# Entry point for running the simulation
# Will orchestrate the simulation runs, parameter loading, and statistics reporting.

import argparse  # For command-line arguments
import csv  # For logging spin results
import logging  # at imports
import os
import random
import statistics  # Added for ROE
import sys  # For redirecting tqdm output
import time
from collections import defaultdict
from datetime import datetime
from typing import Set, Tuple

from simulator import config

# Change relative imports to absolute
from simulator.core.grid import Grid
from simulator.core.state import GameState  # Added GameState import
from simulator.core.symbol import SymbolType

# Joblib/multiprocessing imports deferred until ROE calculations to avoid unnecessary numpy loading


LOG_DIR = "simulation_results"

logger = logging.getLogger(__name__)

# Basic configuration for the logger if this module is run directly
# Other modules using this logger will inherit this if not configured elsewhere
if not logger.hasHandlers():  # Avoid adding multiple handlers if imported
    _handler = logging.StreamHandler(sys.stdout)  # Default to stdout
    _formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    _handler.setFormatter(_formatter)
    logger.addHandler(_handler)
    # Default level, can be overridden by command-line args later
    # or if this module is imported and configured by another entry point.
    logger.setLevel(logging.INFO)


def run_base_game_spin(
    grid: Grid,
    base_bet: float,
    spin_index: int = 0,
    verbose: bool = False,
    debug_rtp: bool = False,
) -> Tuple[float, int]:
    """
    Runs a single complete base game spin sequence (initial drop + avalanches).
    Returns: (total_win_for_spin, total_scatters_landed)
    Added spin_index and verbose flag.
    Removed EW collection tracking as it's FS-specific.
    Added debug_rtp flag for RTP investigation.
    """
    game_state = "BG"
    # Pass spin index for debugging if debug_rtp is enabled
    debug_spin_idx = spin_index if debug_rtp else None
    grid.initialize_spin(game_state, debug_spin_index=debug_spin_idx)

    if verbose:
        logger.info(f"\n--- Spin {spin_index + 1}: Initial Grid --- ")
        grid.display()

    total_win_for_spin = 0.0
    avalanches_in_spin = 0
    scatters_landed_this_spin = grid.count_scatters()
    keep_going = True
    landed_in_previous_step: Set[
        Tuple[int, int]
    ] = set()  # Track coords from previous step

    while keep_going:
        clusters = grid.find_clusters()

        # --- Process Wins (if any clusters found) ---
        if clusters:
            current_multiplier = grid.get_current_multiplier(game_state=game_state)
            win_from_clusters = grid.calculate_win(
                clusters, base_bet, current_multiplier
            )
            total_win_for_spin += win_from_clusters

            if verbose:
                cluster_details = [(s.name, len(c)) for s, c in clusters]
                logger.info(
                    f"Avalanche {avalanches_in_spin}: Win={win_from_clusters:.2f} (Mult:x{current_multiplier}) Clusters: {cluster_details}"
                )
            # Increment multiplier ONLY for winning clusters
            grid.increment_multiplier(game_state=game_state)
        elif debug_rtp:
            logger.debug(
                f"[MAIN-DEBUG] Spin {spin_index + 1}, Avalanche {avalanches_in_spin}: No clusters found"
            )
        # No else needed here, EW explosion handled next

        # --- Process Explosions and Wild Spawning ---
        # Now returns: (cleared_coords, ew_collected_count, did_ew_explode_flag, ew_explosion_details, spawned_wild_coords)
        (
            cleared_coords,
            _ew_collected_this_step,
            did_ew_explode_flag,
            ew_explosion_details,
            spawned_wild_coords,
        ) = grid.process_explosions_and_spawns(clusters, landed_in_previous_step)

        if verbose and did_ew_explode_flag and not clusters:
            logger.info(
                f"Avalanche {avalanches_in_spin}: No winning clusters, but EW exploded."
            )

        # --- Determine if loop should continue based on events THIS cycle ---
        should_process_avalanche = bool(clusters) or did_ew_explode_flag

        if not should_process_avalanche:
            # Nothing happened (no clusters, no explosions), end the loop.
            if verbose:
                logger.info(
                    "No clusters found and no EW explosion occurred. Spin sequence ends."
                )
            keep_going = False
            continue  # Skip avalanche step

        # --- Apply Avalanche (only if clusters existed or EW exploded) ---
        if verbose:
            logger.info(
                "  Grid before avalanche (showing newly spawned wilds marked with +):"
            )
            grid.display(highlight_coords=spawned_wild_coords)

        # Pass the game state to apply_avalanche for proper symbol generation
        _fall_movements, _refill_data, newly_landed = grid.apply_avalanche(game_state)
        made_change = len(newly_landed) > 0
        landed_in_previous_step = newly_landed  # Update for the next iteration

        if verbose:
            logger.info("  Grid after avalanche:")
            grid.display()

        # --- Check for grid stability AFTER avalanche ---
        # If the avalanche didn't move any symbols, the grid is stable.
        if not made_change:
            if verbose:
                logger.info("Avalanche resulted in no change. Spin sequence ends.")
            keep_going = False  # Signal loop to end *after* this iteration completes
            # We don't 'continue' here, still need to count scatters etc for this final state

        # --- Prepare for next iteration (or final state) ---
        avalanches_in_spin += 1
        # Count newly landed scatters
        new_scatters_this_avalanche = 0
        for r_new, c_new in grid.landed_coords:
            symbol = grid._get_symbol(r_new, c_new)
            if symbol and symbol.type == SymbolType.SCATTER:
                new_scatters_this_avalanche += 1
        scatters_landed_this_spin += new_scatters_this_avalanche

        # Safety break
        if avalanches_in_spin > 50:
            logger.warning(
                f"WARN: Spin {spin_index + 1} exceeded 50 avalanches. Breaking loop."
            )
            keep_going = False  # Force loop end

    # Final verbose print - remove EW count
    if verbose:
        logger.info(f"--- Spin {spin_index + 1} Finished --- ")
        logger.info(
            f"Total Win: {total_win_for_spin:.2f}, Avalanches: {avalanches_in_spin}, Total Scatters Seen: {scatters_landed_this_spin}"
        )

    return total_win_for_spin, scatters_landed_this_spin


def calculate_retrigger_spins(scatter_count: int) -> int:
    """Calculates additional spins based on scatter count during FS (GDD 4.8)."""
    if scatter_count < 2:
        return 0
    spins = config.FS_RETRIGGER_SCATTERS.get(scatter_count)
    if spins is None:  # Handle 5+
        max_defined_scatters = max(config.FS_RETRIGGER_SCATTERS.keys())
        spins = (
            config.FS_RETRIGGER_SCATTERS[max_defined_scatters]
            + (scatter_count - max_defined_scatters)
            * config.FS_RETRIGGER_SCATTERS_EXTRA
        )
    return spins


def run_free_spins_feature(
    grid: Grid,
    base_bet: float,
    initial_spins: int,
    trigger_spin_index: int,
    verbose: bool,
) -> float:
    """Runs the complete Free Spins feature session."""
    if verbose:
        logger.info(
            f"\n{'='*15} Free Spins Triggered (BG Spin {trigger_spin_index + 1}) {'='*15}"
        )
        logger.info(f"Initial Spins: {initial_spins}")

    # --- Feature State ---
    remaining_spins = initial_spins
    try:
        current_level_index = config.FS_BASE_MULTIPLIER_LEVELS.index(1)
    except ValueError:
        logger.warning(
            "WARN: FS Base Multiplier level 1 not found! Defaulting index to 0."
        )
        current_level_index = 0
    ew_collected_session = 0
    total_fs_win = 0.0
    total_fs_spins_played = 0
    landed_in_previous_step_fs: Set[
        Tuple[int, int]
    ] = set()  # Track coords from previous step in FS

    # --- Main Free Spins Loop ---
    while remaining_spins > 0:
        total_fs_spins_played += 1
        fs_base_multiplier_level = config.FS_BASE_MULTIPLIER_LEVELS[current_level_index]

        spin_win_this_fs = 0.0
        avalanches_this_fs = 0
        scatters_landed_this_fs_spin = 0
        ew_collected_this_fs_spin = 0
        keep_going_this_spin = True

        if verbose:
            logger.info(
                f"\n--- FS Spin {total_fs_spins_played} (Rem: {remaining_spins-1}) Base Mult: x{fs_base_multiplier_level} EW Coll: {ew_collected_session} ---"
            )

        grid.initialize_spin("FS")
        scatters_landed_this_fs_spin += grid.count_scatters()
        if verbose:
            logger.info("FS Initial Grid:")
            grid.display()

        # --- Avalanche Loop for this Free Spin ---
        while keep_going_this_spin:
            clusters = grid.find_clusters()

            if clusters:
                current_fs_multiplier = grid.get_current_multiplier(
                    game_state="FS",
                    fs_base_multiplier=fs_base_multiplier_level,
                    fs_avalanche_count=avalanches_this_fs,
                )
                win_from_clusters = grid.calculate_win(
                    clusters, base_bet, current_fs_multiplier
                )
                spin_win_this_fs += win_from_clusters
                if verbose:
                    cluster_details = [(s.name, len(c)) for s, c in clusters]
                    logger.info(
                        f"  FS Avalanche {avalanches_this_fs}: Win={win_from_clusters:.2f} (Mult:x{current_fs_multiplier}) Clusters: {cluster_details}"
                    )

            # grid.process_explosions_and_spawns returns 5 values:
            # (cleared_coords, ew_collected_count, did_ew_explode_flag, ew_explosion_details, spawned_wild_coords)
            (
                cleared_coords,
                ew_collected_this_step,
                did_ew_explode_flag,
                ew_explosion_details,
                spawned_wild_coords,
            ) = grid.process_explosions_and_spawns(clusters, landed_in_previous_step_fs)
            ew_collected_this_fs_spin += ew_collected_this_step
            if verbose and ew_collected_this_step > 0:
                logger.info(
                    f"    Collected {ew_collected_this_step} EWs this avalanche step."
                )
            if verbose and did_ew_explode_flag and not clusters:
                logger.info(
                    f"  FS Avalanche {avalanches_this_fs}: No winning clusters, but EW exploded."
                )

            # --- Determine if loop should continue based on events THIS cycle ---
            should_process_avalanche = bool(clusters) or did_ew_explode_flag

            if not should_process_avalanche:
                # Nothing happened (no clusters, no explosions), end the loop for this FS spin.
                if verbose:
                    logger.info(
                        "  FS: No clusters or EW explosion. Ending avalanche sequence for this FS spin."
                    )
                keep_going_this_spin = False
                continue  # Skip avalanche step

            # --- Apply Avalanche (only if clusters existed or EW exploded) ---
            if verbose:
                logger.info("    Grid before FS avalanche:")
                grid.display(highlight_coords=spawned_wild_coords)

            # Returns tuple: (fall_movements, refill_data, newly_landed_coords)
            _fall_movements_fs, _refill_data_fs, newly_landed_fs = grid.apply_avalanche(
                "FS"
            )
            made_change = len(newly_landed_fs) > 0
            landed_in_previous_step_fs = (
                newly_landed_fs  # Update for the next iteration
            )

            if verbose:
                logger.info("    Grid after FS avalanche:")
                grid.display()

            # --- Check for grid stability AFTER avalanche ---
            # If the avalanche didn't move any symbols, the grid is stable.
            if not made_change:
                if verbose:
                    logger.info(
                        "  FS: Avalanche resulted in no change. Ending avalanche sequence."
                    )
                keep_going_this_spin = (
                    False  # Signal loop to end *after* this iteration
                )
                # Don't continue, let iteration finish

            # --- Prepare for next iteration (or final state) ---
            avalanches_this_fs += 1
            # Count newly landed scatters
            new_scatters_this_avalanche = 0
            for r_new, c_new in grid.landed_coords:
                symbol = grid._get_symbol(r_new, c_new)
                if symbol and symbol.type == SymbolType.SCATTER:
                    new_scatters_this_avalanche += 1
            scatters_landed_this_fs_spin += new_scatters_this_avalanche

            # Safety break
            if avalanches_this_fs > 50:
                logger.warning(
                    f"WARN: FS Spin {total_fs_spins_played} exceeded 50 avalanches. Breaking loop."
                )
                keep_going_this_spin = False  # Force loop end
        # --- End Avalanche Loop ---

        # --- Post-Spin Processing ---
        remaining_spins -= 1
        total_fs_win += spin_win_this_fs

        # --- ADDED: Max Win Cap Check ---
        max_win_cap = base_bet * 7500
        if total_fs_win >= max_win_cap:
            capped_win = max_win_cap
            if verbose:
                logger.info(
                    f"!!! FS Max Win Cap Reached! Capping win at {capped_win:.2f} (was {total_fs_win:.2f}) !!!",
                    flush=True,
                )
            total_fs_win = capped_win
            remaining_spins = 0  # End the feature
            # We break here to prevent further processing like retriggers/upgrades after hitting the cap
            break
        # --- End Max Win Cap Check ---

        # Check for retriggers only if cap not hit
        if scatters_landed_this_fs_spin >= 2:
            additional_spins = calculate_retrigger_spins(scatters_landed_this_fs_spin)
            if additional_spins > 0:
                remaining_spins += additional_spins
                if verbose:
                    logger.info(
                        f"  Retrigger! +{additional_spins} spins (Scatters: {scatters_landed_this_fs_spin}). New Remaining: {remaining_spins}"
                    )

        # Check for upgrades only if cap not hit
        ew_collected_session += ew_collected_this_fs_spin
        if ew_collected_session >= config.FS_EW_COLLECTION_PER_UPGRADE:
            pending_upgrades = (
                ew_collected_session // config.FS_EW_COLLECTION_PER_UPGRADE
            )
            if pending_upgrades > 0:
                ew_collected_session -= (
                    pending_upgrades * config.FS_EW_COLLECTION_PER_UPGRADE
                )
                spins_from_upgrade = pending_upgrades * config.FS_SPINS_PER_UPGRADE
                remaining_spins += spins_from_upgrade
                level_before = config.FS_BASE_MULTIPLIER_LEVELS[current_level_index]
                # Update index for the *next* spin
                current_level_index = min(
                    current_level_index + pending_upgrades,
                    len(config.FS_BASE_MULTIPLIER_LEVELS) - 1,
                )
                level_after = config.FS_BASE_MULTIPLIER_LEVELS[current_level_index]
                if verbose:
                    logger.info(
                        f"  Multiplier Upgrade! {pending_upgrades} level(s). Base Mult x{level_before} -> x{level_after}. Spins Added: +{spins_from_upgrade}. New Remaining: {remaining_spins}. EW Counter now: {ew_collected_session}."
                    )

        if verbose:
            logger.info(
                f"--- End FS Spin {total_fs_spins_played} --- Spin Win: {spin_win_this_fs:.2f}, Total FS Win: {total_fs_win:.2f} ---"
            )

    # --- End of Feature ---
    if verbose:
        logger.info(f"{'='*15} Free Spins Finished {'='*15}")
        logger.info(f"Total FS Win: {total_fs_win:.2f}")

    return total_fs_win


def calculate_roe(
    rtp: float,
    base_bet_for_sim: float,
    roe_bet: float = 1.0,
    num_roe_sims: int = 1000,
    max_roe_spins: int = 1_000_000,
    *,
    base_seed: int | None = None,
) -> Tuple[str, str]:
    """
    Calculates Median and Average Rate of Exhaustion (ROE) in parallel.

    Runs multiple simulations starting with a balance of 100x roe_bet,
    counting spins (N) until the balance drops below roe_bet.

    Args:
        rtp: The overall Return to Player (%) calculated from the main simulation.
        base_bet_for_sim: The base bet used in the main simulation (needed for context, though ROE uses roe_bet).
        roe_bet: The bet amount used for ROE simulations (default 1.0).
        num_roe_sims: The number of ROE simulations to run (default 1000).
        max_roe_spins: The maximum number of spins per ROE simulation before considering it infinite (default 1,000,000).
        base_seed: Optional base seed for deterministic parallel runs

    Returns:
        A tuple containing (Median ROE, Average ROE) as strings (can be "Infinite").
    """
    # Import necessary modules here to keep initial load light
    # import multiprocessing  # Keep for cpu_count if not using joblib's one, or for other context F401 unused

    from joblib import Parallel, cpu_count, delayed
    from tqdm import tqdm

    from simulator.core.rng import SpinRNG  # For per-worker RNG in ROE

    if rtp >= 100.0:
        logger.info("Calculating ROE: RTP >= 100%, ROE is considered Infinite.")
        return "Infinite", "Infinite"
    if num_roe_sims <= 0 or max_roe_spins <= 0:
        logger.error(
            f"Calculating ROE: Invalid parameters (num_roe_sims={num_roe_sims}, max_roe_spins={max_roe_spins}). ROE Error."
        )
        return "Error", "Error"

    start_balance = roe_bet * 100  # Example: 100 bets
    roe_sim_start_time = time.time()

    # --- Helper function for a single ROE simulation ---
    # This helper needs to simulate spins until balance is exhausted or max_roe_spins is hit.
    # It should use its own Grid and GameState, seeded appropriately.
    def _run_single_roe_sim(sim_idx: int) -> float:  # Removed callback parameter
        worker_seed = (base_seed + sim_idx) if base_seed is not None else None
        worker_rng = SpinRNG(seed=worker_seed)  # Each ROE sim gets its own RNG stream

        # Create a new Grid for each ROE simulation, using the worker_rng
        # GameState is simple and can be created fresh.
        roe_game_state = GameState()  # Create a GameState for the ROE grid
        roe_grid = Grid(
            state=roe_game_state,
            rows=config.GRID_ROWS,
            cols=config.GRID_COLS,
            # SYMBOLS, WEIGHTS_BG, WEIGHTS_FS are implicitly handled by Grid via config
            rng=worker_rng,
        )

        balance = start_balance
        n_spins = 0

        while balance >= roe_bet:  # Continue as long as balance can cover one more bet
            if n_spins >= max_roe_spins:
                return float("inf")  # Signal infinite run for this simulation

            balance -= roe_bet  # Place the bet
            n_spins += 1

            # Run a single base game spin and potential subsequent FS
            # These calls should NOT be verbose for ROE calculations
            spin_win_bg, scatters_in_seq = run_base_game_spin(
                roe_grid,
                roe_bet,
                spin_index=n_spins - 1,
                verbose=False,
                debug_rtp=False,
            )
            current_round_total_win = spin_win_bg

            if scatters_in_seq >= 3:
                initial_fs = config.FS_TRIGGER_SCATTERS.get(scatters_in_seq)
                if initial_fs is None:  # Handle 5+ scatters case
                    max_sc_defined = max(
                        k
                        for k in config.FS_TRIGGER_SCATTERS.keys()
                        if isinstance(k, int)
                    )
                    initial_fs = (
                        config.FS_TRIGGER_SCATTERS[max_sc_defined]
                        + (scatters_in_seq - max_sc_defined)
                        * config.FS_TRIGGER_SCATTERS_EXTRA
                    )

                win_from_fs = run_free_spins_feature(
                    roe_grid,
                    roe_bet,
                    initial_fs,
                    trigger_spin_index=n_spins - 1,
                    verbose=False,
                )
                current_round_total_win += win_from_fs

            balance += current_round_total_win  # Add winnings from the round

        return float(n_spins)  # Return number of spins until exhaustion

    # --- End of helper function ---

    num_cores = cpu_count()
    logger.info(
        f"Calculating ROE: {num_roe_sims} sims, max {max_roe_spins} spins/sim, bet={roe_bet:.2f}, using {num_cores} cores."
    )

    spins_to_exhaustion_results = []
    infinite_roe_count = 0

    # Create a list of tasks for Parallel
    tasks = [delayed(_run_single_roe_sim)(i) for i in range(num_roe_sims)]

    # Run in parallel and wrap with tqdm for progress in the main process
    # This pattern is generally safer for pickling with tqdm.
    # tqdm will iterate over the results as they are completed if Parallel is used as a context manager or its result directly iterated.
    results = []
    with tqdm(
        total=len(tasks), desc="ROE Simulations", unit="sim", file=sys.stderr
    ) as pbar:
        # Use dispatch_callback for tqdm updates if available and preferred for joblib
        # A simpler way if Parallel returns an iterable that completes:
        parallel_results = Parallel(n_jobs=num_cores)(tasks)
        for res_item in parallel_results:
            results.append(res_item)
            pbar.update(1)  # Manually update tqdm after each result is processed

    for res in results:
        if res == float("inf"):
            infinite_roe_count += 1
        elif res is not None:  # Ensure results are valid numbers
            spins_to_exhaustion_results.append(res)

    roe_calc_duration = time.time() - roe_sim_start_time

    if infinite_roe_count == num_roe_sims:  # All simulations hit max_roe_spins
        logger.info(
            f"ROE: All {num_roe_sims} simulations reached {max_roe_spins:,} spins. ROE is effectively Infinite. Time: {roe_calc_duration:.2f}s."
        )
        return "Infinite", "Infinite"
    elif infinite_roe_count > 0:
        logger.info(
            f"ROE: {infinite_roe_count} out of {num_roe_sims} simulations reached {max_roe_spins:,} spins (treated as censored data for median/avg if applicable, or infinite if all)."
        )

    if (
        not spins_to_exhaustion_results
    ):  # No sims ruined, or all were infinite and handled above
        if infinite_roe_count > 0:  # This means all non-error runs were infinite
            logger.info(
                f"ROE: All non-error simulations ({infinite_roe_count}) reached max spins. Effectively infinite. Time: {roe_calc_duration:.2f}s."
            )
            return (
                "Infinite (all reached max spins)",
                "Infinite (all reached max spins)",
            )
        logger.info(
            f"ROE Calculation: No simulations resulted in ruin before {max_roe_spins} spins or all failed. Time: {roe_calc_duration:.2f}s."
        )
        return "Error or All Infinite", "Error or All Infinite"

    median_roe = statistics.median(spins_to_exhaustion_results)
    avg_roe = statistics.mean(spins_to_exhaustion_results)

    logger.info(
        f"ROE Calculation Complete. Median ROE: {median_roe:.2f} spins, Average ROE: {avg_roe:.2f} spins (among those that ruined). Affected by {infinite_roe_count} infinite runs. Time: {roe_calc_duration:.2f}s."
    )
    return f"{median_roe:.2f}", f"{avg_roe:.2f}"


def run_simulation(
    num_spins: int,
    base_bet: float,
    run_id: str,
    verbose_spins: int = 0,
    verbose_fs_only: bool = False,
    return_stats: bool = False,
    enhanced_stats: bool = False,
    seed: int = None,
    output_path_override: str = None,  # For optimizer
    calc_roe_flag: bool = True,
    base_seed_roe: int = None,
):
    from tqdm import tqdm  # tqdm for the main simulation loop

    start_time = time.time()

    # --- RNG Setup ---
    if seed is not None:
        logger.info(f"Seeding main simulation RNG with: {seed}")
        main_rng = random.Random(seed)
    else:
        logger.info("No seed provided, using system time for main simulation RNG.")
        main_rng = random.Random()

    # Create a GameState instance (it's simple and doesn't take args in its current form)
    game_state_instance = GameState()

    # Pass the GameState instance and the seeded RNG to Grid
    grid = Grid(
        state=game_state_instance,
        rows=config.GRID_ROWS,
        cols=config.GRID_COLS,
        rng=main_rng,
    )

    # --- CSV Logging Setup ---
    log_directory = output_path_override if output_path_override else LOG_DIR
    os.makedirs(log_directory, exist_ok=True)
    spin_log_filename = os.path.join(log_directory, f"{run_id}_spin_log.csv")
    logger.info(f"Output CSV log: {spin_log_filename}")

    csv_file_handle = None
    csv_writer = None
    try:
        csv_file_handle = open(spin_log_filename, "w", newline="")
        csv_writer = csv.writer(csv_file_handle)
        csv_writer.writerow(
            [
                "Spin_Number",
                "Spin_Type",
                "Spin_Win",
                "Scatters_Landed",
                "Initial_FS_Spins",
                "FS_Win",
                "FS_Spins_Played",
                "FS_EW_Collected",
                "FS_Retrigger_Spins_Awarded",
                "Total_Win_After_Spin",
                "Final_Multiplier_Level",
            ]
        )

        # --- Stats Tracking (initialize before loop) ---
        total_win = 0.0
        total_bg_win = 0.0
        fs_total_win = 0.0
        hit_count = 0
        fs_triggers = 0
        total_scatters_seen = 0
        max_win_overall = 0.0
        max_win_multiplier = 0.0
        # For enhanced stats (can be expanded)
        win_distribution = defaultdict(int)  # Tracks frequency of win multipliers

        # --- Simulation Start ---
        logger.info(f"Starting simulation: {run_id}")
        logger.info(f"Number of spins: {num_spins:,}, Base bet: {base_bet:.2f}")
        if verbose_spins > 0:
            logger.info(f"Verbose logging for the first {verbose_spins} spins.")
        if verbose_fs_only:
            logger.info("Verbose logging for Free Spins feature.")

        # --- Main Simulation Loop ---
        progress_bar = tqdm(
            range(num_spins),
            desc=f"Simulating {run_id}",
            unit="spin",
            disable=return_stats,
            file=sys.stderr,
        )

        for i in progress_bar:
            verbose_this_spin = (i < verbose_spins) and not verbose_fs_only
            debug_rtp_this_spin = (
                False  # Default, can be changed for specific debugging
            )

            # --- Run Base Game Spin ---
            spin_win_bg, scatters_this_spin_seq = run_base_game_spin(
                grid,
                base_bet,
                spin_index=i,
                verbose=verbose_this_spin,
                debug_rtp=debug_rtp_this_spin,
            )

            current_round_total_win = spin_win_bg  # Win from BG part
            fs_win_this_round = 0.0  # Specific to FS part of this round
            initial_fs_spins_awarded = 0
            fs_spins_played_this_round = 0
            fs_ew_collected_this_round = (
                0  # Need to get this from run_free_spins_feature if desired for CSV
            )
            fs_retrigger_spins_this_round = (
                0  # Need to get this from run_free_spins_feature
            )

            total_scatters_seen += scatters_this_spin_seq

            # --- Free Spins Feature Trigger ---
            fs_triggered_this_spin = False
            if scatters_this_spin_seq >= 3:
                fs_triggered_this_spin = True
                fs_triggers += 1
                initial_fs_spins_awarded = config.FS_TRIGGER_SCATTERS.get(
                    scatters_this_spin_seq
                )
                if initial_fs_spins_awarded is None:
                    max_sc_defined = max(
                        k
                        for k in config.FS_TRIGGER_SCATTERS.keys()
                        if isinstance(k, int)
                    )
                    initial_fs_spins_awarded = (
                        config.FS_TRIGGER_SCATTERS[max_sc_defined]
                        + (scatters_this_spin_seq - max_sc_defined)
                        * config.FS_TRIGGER_SCATTERS_EXTRA
                    )

                verbose_this_fs = verbose_fs_only or verbose_this_spin
                # run_free_spins_feature needs to return more detailed stats if we want to log them in CSV for FS
                # For now, it only returns fs_win. Let's assume we might extend it later.
                fs_win_this_round = run_free_spins_feature(
                    grid,
                    base_bet,
                    initial_fs_spins_awarded,
                    trigger_spin_index=i,
                    verbose=verbose_this_fs,
                )
                current_round_total_win += fs_win_this_round

            # --- Max Win Cap (applied to total win for the game round) ---
            max_win_cap_val = base_bet * config.MAX_WIN_CAP_MULTIPLIER
            if current_round_total_win > max_win_cap_val:
                # Log capping if verbose
                if verbose_this_spin or (verbose_fs_only and fs_triggered_this_spin):
                    logger.info(
                        f"Round {i+1} total win {current_round_total_win:.2f} (BG:{spin_win_bg:.2f}, FS:{fs_win_this_round:.2f}) capped to {max_win_cap_val:.2f}"
                    )
                current_round_total_win = max_win_cap_val
                # If capped, need to decide how to attribute fs_win_this_round vs spin_win_bg for stats.
                # Simplest: keep original components for separate BG/FS tracking, but total_win reflects cap.
                # Or, attribute cap proportionally. For now, fs_total_win and total_bg_win track pre-cap components.

            # --- Accumulate Stats ---
            total_win += current_round_total_win
            if fs_triggered_this_spin:
                fs_total_win += (
                    fs_win_this_round  # fs_win_this_round is pre-overall-cap component
                )
            else:  # Win is purely from BG
                total_bg_win += spin_win_bg  # spin_win_bg is pre-overall-cap component

            if current_round_total_win > 0:
                hit_count += 1

            current_win_multiplier = (
                current_round_total_win / base_bet if base_bet > 0 else 0
            )
            if current_round_total_win > max_win_overall:
                max_win_overall = current_round_total_win
                max_win_multiplier = current_win_multiplier

            if enhanced_stats:
                win_distribution[round(current_win_multiplier)] += 1

            # --- Write to CSV ---
            if csv_writer:
                csv_writer.writerow(
                    [
                        i + 1,
                        "FS_TRIGGER" if fs_triggered_this_spin else "BG",
                        f"{spin_win_bg:.2f}",  # BG component of the win
                        scatters_this_spin_seq,
                        initial_fs_spins_awarded,
                        f"{fs_win_this_round:.2f}",  # FS component of the win
                        fs_spins_played_this_round,  # Placeholder - needs value from run_free_spins_feature
                        fs_ew_collected_this_round,  # Placeholder
                        fs_retrigger_spins_this_round,  # Placeholder
                        f"{current_round_total_win:.2f}",  # Total win for the round (potentially capped)
                        grid.get_current_multiplier(
                            "FS" if fs_triggered_this_spin else "BG"
                        ),  # Multiplier state at end of round
                    ]
                )

        progress_bar.close()
        logger.info("Simulation calculations complete. Generating summary...")

        # --- Final Stats Calculation ---
        end_time = time.time()
        simulation_duration = end_time - start_time
        total_staked = num_spins * base_bet
        rtp_final = (total_win / total_staked) * 100 if total_staked > 0 else 0
        hit_freq_final = (hit_count / num_spins) * 100 if num_spins > 0 else 0

        # --- Output Summary (to logger) ---
        summary_lines = []
        summary_lines.append("\n--- Simulation Summary ---")
        summary_lines.append(f"Run ID: {run_id}")
        summary_lines.append(
            f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        summary_lines.append(f"Total Spins: {num_spins:,}")
        summary_lines.append(f"Base Bet: {base_bet:.2f}")
        summary_lines.append(f"Total Staked: {total_staked:,.2f}")
        summary_lines.append(
            f"Total Win (Capped): {total_win:,.2f}"
        )  # Clarify this is overall capped win
        summary_lines.append(
            f"  Base Game Win (Component): {total_bg_win:,.2f}"
        )  # Pre-overall-cap component
        summary_lines.append(
            f"  Free Spins Win (Component): {fs_total_win:,.2f}"
        )  # Pre-overall-cap component
        summary_lines.append(f"Return to Player (RTP): {rtp_final:.4f}%")

        if calc_roe_flag:
            roe_seed_to_use = base_seed_roe if base_seed_roe is not None else seed
            if roe_seed_to_use is not None:
                logger.info(f"Using base_seed {roe_seed_to_use} for ROE calculations.")
            else:
                logger.info(
                    "No base_seed for ROE, will be non-deterministic for ROE if run in parallel across calls."
                )

            # Call the refactored calculate_roe
            median_roe_str, avg_roe_str = calculate_roe(
                rtp=rtp_final,
                base_bet_for_sim=base_bet,
                roe_bet=1.0,
                num_roe_sims=config.ROE_NUM_SIMULATIONS,  # Use config value
                max_roe_spins=config.ROE_MAX_SPINS,  # Use config value
                base_seed=roe_seed_to_use,
            )
            summary_lines.append(f"Median ROE: {median_roe_str} spins (at 1.0 bet)")
            summary_lines.append(f"Average ROE: {avg_roe_str} spins (at 1.0 bet)")
        else:
            summary_lines.append("ROE Calculation: Skipped")

        summary_lines.append(f"\nHit Count: {hit_count:,}")
        summary_lines.append(f"Hit Frequency: {hit_freq_final:.2f}%")
        summary_lines.append(
            f"\nTotal Scatters Seen (in sequences): {total_scatters_seen:,}"
        )
        summary_lines.append(f"Free Spins Triggers (>=3 Scatters): {fs_triggers:,}")
        if num_spins > 0 and fs_triggers > 0:
            fs_trigger_freq = (fs_triggers / num_spins) * 100
            fs_trigger_rate = num_spins / fs_triggers
            summary_lines.append(f"  FS Trigger Frequency: {fs_trigger_freq:.4f}%")
            summary_lines.append(
                f"  FS Trigger Rate: ~1 in {fs_trigger_rate:.1f} spins"
            )
        else:
            summary_lines.append("  FS Trigger Frequency: 0.0000%")
            summary_lines.append("  FS Trigger Rate: N/A")
            fs_trigger_freq = 0

        summary_lines.append(f"\nSimulation Time: {simulation_duration:.2f} seconds")
        summary_lines.append(
            f"Spins per second: {(num_spins / simulation_duration):.2f}"
            if simulation_duration > 0
            else "N/A"
        )

        if enhanced_stats:
            avg_win_per_fs_trigger = (
                (fs_total_win / fs_triggers) if fs_triggers > 0 else 0
            )
            summary_lines.append(
                f"\n  Avg Win per FS Trigger (Component): {avg_win_per_fs_trigger:,.2f}"
            )
            summary_lines.append(
                f"Max Win (Overall): {max_win_overall:,.2f} ({max_win_multiplier:,.2f}x)"
            )
            # Add win distribution summary if desired
            # summary_lines.append("\nWin Distribution (Multiplier: Count):")
            # for mult, count in sorted(win_distribution.items()):
            #     summary_lines.append(f"  x{mult}: {count}")

        for line in summary_lines:
            logger.info(line)

        if return_stats:
            stats_dict = {
                "run_id": run_id,
                "total_spins": num_spins,
                "base_bet": base_bet,
                "total_staked": total_staked,
                "total_win": total_win,
                "rtp": rtp_final,
                "total_bg_win": total_bg_win,
                "fs_total_win": fs_total_win,
                "hit_count": hit_count,
                "hit_frequency": hit_freq_final,
                "fs_triggers": fs_triggers,
                "fs_trigger_frequency": (
                    fs_trigger_freq if (num_spins > 0 and fs_triggers > 0) else 0
                ),
                "total_scatters_seen": total_scatters_seen,
                "simulation_duration_seconds": simulation_duration,
                "spins_per_second": (
                    (num_spins / simulation_duration) if simulation_duration > 0 else 0
                ),
                "max_win": max_win_overall,
                "max_win_multiplier": max_win_multiplier,
            }
            if calc_roe_flag:
                stats_dict["median_roe"] = median_roe_str
                stats_dict["average_roe"] = avg_roe_str
            if enhanced_stats:
                stats_dict["avg_win_per_fs_trigger"] = (
                    avg_win_per_fs_trigger if fs_triggers > 0 else 0
                )
                stats_dict["win_distribution"] = dict(win_distribution)
            return stats_dict

    except Exception as e:
        logger.error(f"Error during simulation {run_id}: {e}", exc_info=True)
        if return_stats:  # Return partial or error stats if an exception occurs
            return {
                "error": str(e),
                "run_id": run_id,
                "total_spins_completed": i if "i" in locals() else 0,
            }
        # If not returning stats, the error is logged, and the function will implicitly return None
    finally:
        if csv_file_handle:
            csv_file_handle.close()
            logger.info(f"Spin details CSV closed: {spin_log_filename}")

    # Implicit None return if not return_stats and no error before this point,
    # or if an error occurred and not return_stats.


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Esqueleto Explosivo 3 Simulator")
    parser.add_argument(
        "--spins", type=int, default=1000, help="Number of spins to simulate"
    )
    parser.add_argument("--bet", type=float, default=1.0, help="Base bet amount")
    parser.add_argument(
        "--verbose",
        "-v",
        type=int,
        default=0,
        metavar="N",
        help="Show detailed logs for the first N spins.",
    )
    parser.add_argument(
        "--verbose-fs",
        action="store_true",
        help="Show detailed logs for Free Spins feature only.",
    )
    parser.add_argument(
        "--id",
        type=str,
        default="sim_run",
        help="ID for the simulation run (affects output filenames)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Seed for the random number generator for reproducible runs.",
    )
    parser.add_argument(
        "--enhanced-stats",
        action="store_true",
        help="Calculate and show more detailed statistics.",
    )
    # ROE specific args
    parser.add_argument("--no-roe", action="store_true", help="Skip ROE calculation.")
    # JIT args
    parser.add_argument(
        "--jit",
        action="store_true",
        default=None,
        help="Enable JIT compilation (if Numba is available).",
    )
    parser.add_argument(
        "--no-jit", action="store_false", dest="jit", help="Disable JIT compilation."
    )
    # Log level
    parser.add_argument(
        "--log",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level.",
    )

    args = parser.parse_args()

    # --- Setup Logging Handler and Level based on args ---
    # Remove existing handlers if any were added by basicConfig earlier for library use
    # Or, ensure the initial logger setup is only done here.
    # For simplicity, let's assume the initial setup is good as a default,
    # and here we potentially override the level or add specific handlers if needed.

    # Reconfigure logger based on --log argument
    # Get the root logger if configuring globally, or the specific 'logger' instance
    # For now, configuring the 'logger' instance used in this file.
    # If other modules also log, they might need their own setup or inherit.

    log_level_str = args.log.upper()
    numeric_level = getattr(logging, log_level_str, None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {log_level_str}")

    # Clear any handlers already set on this specific logger instance
    # This is to ensure that if this script is re-run or function called multiple times in a session,
    # we don't keep adding handlers.
    # A more robust setup might involve a central logging config.
    if logger.hasHandlers():
        logger.handlers.clear()

    # Add a new handler with the specified level
    # This ensures that the command-line --log flag correctly sets the level.
    ch = logging.StreamHandler(
        sys.stdout
    )  # Ensure tqdm and logging don't fight over stderr/stdout
    ch.setLevel(numeric_level)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logger.setLevel(numeric_level)  # Set level on the logger itself too

    # Set global JIT flag
    if args.jit is not None:
        config.ENABLE_JIT = args.jit
        logger.info(f"JIT compilation explicitly set to: {config.ENABLE_JIT}")
    else:
        # Default behavior: use JIT if available, unless overridden
        config.ENABLE_JIT = config.JIT_AVAILABLE
        logger.info(
            f"JIT compilation default: {config.ENABLE_JIT} (available: {config.JIT_AVAILABLE})"
        )

    logger.info(f"Running simulation with ID: {args.id}")

    run_simulation(
        num_spins=args.spins,
        base_bet=args.bet,
        run_id=args.id,
        verbose_spins=args.verbose,
        verbose_fs_only=args.verbose_fs,
        seed=args.seed,
        enhanced_stats=args.enhanced_stats,
        calc_roe_flag=not args.no_roe,  # Pass the flag
        base_seed_roe=args.seed,  # Pass main seed as base for ROE seed derivation
    )
