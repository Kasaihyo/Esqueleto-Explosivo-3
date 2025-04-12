# Entry point for running the simulation
# Will orchestrate the simulation runs, parameter loading, and statistics reporting.

import time
import random
from collections import defaultdict
from typing import Tuple
import argparse # For command-line arguments
import csv # For logging spin results
import os
from datetime import datetime

# Change relative imports to absolute
from simulator.core.grid import Grid
from simulator.core.symbol import SymbolType
from simulator import config

LOG_DIR = "simulation_results"

def run_base_game_spin(grid: Grid, base_bet: float, spin_index: int = 0, verbose: bool = False) -> Tuple[float, int]:
    """
    Runs a single complete base game spin sequence (initial drop + avalanches).
    Returns: (total_win_for_spin, total_scatters_landed)
    Added spin_index and verbose flag.
    Removed EW collection tracking as it's FS-specific.
    """
    game_state = "BG"
    grid.initialize_spin(game_state)
    if verbose:
        print(f"\n--- Spin {spin_index + 1}: Initial Grid --- ")
        grid.display()

    total_win_for_spin = 0.0
    avalanches_in_spin = 0
    scatters_landed_this_spin = grid.count_scatters()
    keep_going = True

    while keep_going:
        clusters = grid.find_clusters()

        # --- Process Wins (if any clusters found) ---
        if clusters:
            current_multiplier = grid.get_current_multiplier(game_state=game_state)
            win_from_clusters = grid.calculate_win(clusters, base_bet, current_multiplier)
            total_win_for_spin += win_from_clusters
            if verbose:
                cluster_details = [(s.name, len(c)) for s,c in clusters]
                print(f"Avalanche {avalanches_in_spin}: Win={win_from_clusters:.2f} (Mult:x{current_multiplier}) Clusters: {cluster_details}")
            # Increment multiplier ONLY for winning clusters
            grid.increment_multiplier(game_state=game_state)
        # No else needed here, EW explosion handled next

        # --- Process Explosions and Wild Spawning ---
        # Now returns: (cleared_coords, ew_collected_count, did_ew_explode_flag, spawned_wild_coords)
        cleared_coords, _ew_collected_this_step, did_ew_explode_flag, spawned_wild_coords = grid.process_explosions_and_spawns(clusters)
        if verbose and did_ew_explode_flag and not clusters:
             print(f"Avalanche {avalanches_in_spin}: No winning clusters, but EW exploded.")

        # --- Determine if loop should continue based on events THIS cycle ---
        should_process_avalanche = bool(clusters) or did_ew_explode_flag

        if not should_process_avalanche:
            # Nothing happened (no clusters, no explosions), end the loop.
            if verbose: print("No clusters found and no EW explosion occurred. Spin sequence ends.")
            keep_going = False
            continue # Skip avalanche step

        # --- Apply Avalanche (only if clusters existed or EW exploded) ---
        if verbose:
            print("  Grid before avalanche (showing newly spawned wilds marked with +):")
            grid.display(highlight_coords=spawned_wild_coords)

        # Pass the game state to apply_avalanche for proper symbol generation
        made_change = len(grid.apply_avalanche(game_state)) > 0

        if verbose:
            print("  Grid after avalanche:")
            grid.display()

        # --- Check for grid stability AFTER avalanche ---
        # If the avalanche didn't move any symbols, the grid is stable.
        if not made_change:
            if verbose: print("Avalanche resulted in no change. Spin sequence ends.")
            keep_going = False # Signal loop to end *after* this iteration completes
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
            print(f"WARN: Spin {spin_index + 1} exceeded 50 avalanches. Breaking loop.")
            keep_going = False # Force loop end

    # Final verbose print - remove EW count
    if verbose:
        print(f"--- Spin {spin_index + 1} Finished --- ")
        print(f"Total Win: {total_win_for_spin:.2f}, Avalanches: {avalanches_in_spin}, Total Scatters Seen: {scatters_landed_this_spin}")

    return total_win_for_spin, scatters_landed_this_spin

def calculate_retrigger_spins(scatter_count: int) -> int:
    """Calculates additional spins based on scatter count during FS (GDD 4.8)."""
    if scatter_count < 2: return 0
    spins = config.FS_RETRIGGER_SCATTERS.get(scatter_count)
    if spins is None: # Handle 5+
        max_defined_scatters = max(config.FS_RETRIGGER_SCATTERS.keys())
        spins = config.FS_RETRIGGER_SCATTERS[max_defined_scatters] + \
                (scatter_count - max_defined_scatters) * config.FS_RETRIGGER_SCATTERS_EXTRA
    return spins

def run_free_spins_feature(grid: Grid, base_bet: float, initial_spins: int, trigger_spin_index: int, verbose: bool) -> float:
    """Runs the complete Free Spins feature session."""
    if verbose:
        print(f"\n{'='*15} Free Spins Triggered (BG Spin {trigger_spin_index + 1}) {'='*15}")
        print(f"Initial Spins: {initial_spins}")

    # --- Feature State ---
    remaining_spins = initial_spins
    try:
        current_level_index = config.FS_BASE_MULTIPLIER_LEVELS.index(1)
    except ValueError:
        print("WARN: FS Base Multiplier level 1 not found! Defaulting index to 0.")
        current_level_index = 0
    ew_collected_session = 0
    total_fs_win = 0.0
    total_fs_spins_played = 0

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
            print(f"\n--- FS Spin {total_fs_spins_played} (Rem: {remaining_spins-1}) Base Mult: x{fs_base_multiplier_level} EW Coll: {ew_collected_session} ---")

        grid.initialize_spin("FS")
        scatters_landed_this_fs_spin += grid.count_scatters()
        if verbose:
            print("FS Initial Grid:")
            grid.display()

        # --- Avalanche Loop for this Free Spin ---
        while keep_going_this_spin:
            clusters = grid.find_clusters()

            if clusters:
                current_fs_multiplier = grid.get_current_multiplier(game_state="FS",
                                                                      fs_base_multiplier=fs_base_multiplier_level,
                                                                      fs_avalanche_count=avalanches_this_fs)
                win_from_clusters = grid.calculate_win(clusters, base_bet, current_fs_multiplier)
                spin_win_this_fs += win_from_clusters
                if verbose:
                    cluster_details = [(s.name, len(c)) for s, c in clusters]
                    print(f"  FS Avalanche {avalanches_this_fs}: Win={win_from_clusters:.2f} (Mult:x{current_fs_multiplier}) Clusters: {cluster_details}")

            cleared_coords, ew_collected_this_step, did_ew_explode_flag, spawned_wild_coords = grid.process_explosions_and_spawns(clusters)
            ew_collected_this_fs_spin += ew_collected_this_step
            if verbose and ew_collected_this_step > 0:
                print(f"    Collected {ew_collected_this_step} EWs this avalanche step.")
            if verbose and did_ew_explode_flag and not clusters:
                print(f"  FS Avalanche {avalanches_this_fs}: No winning clusters, but EW exploded.")

            # --- Determine if loop should continue based on events THIS cycle ---
            should_process_avalanche = bool(clusters) or did_ew_explode_flag

            if not should_process_avalanche:
                # Nothing happened (no clusters, no explosions), end the loop for this FS spin.
                keep_going_this_spin = False
                continue # Skip avalanche step

            # --- Apply Avalanche (only if clusters existed or EW exploded) ---
            if verbose:
                print("    Grid before FS avalanche:")
                grid.display(highlight_coords=spawned_wild_coords)

            # Returns landed coordinates - if any, then changes were made
            made_change = len(grid.apply_avalanche("FS")) > 0

            if verbose:
                print("    Grid after FS avalanche:")
                grid.display()

            # --- Check for grid stability AFTER avalanche ---
            # If the avalanche didn't move any symbols, the grid is stable.
            if not made_change:
                keep_going_this_spin = False # Signal loop to end *after* this iteration
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
                print(f"WARN: FS Spin {total_fs_spins_played} exceeded 50 avalanches. Breaking loop.")
                keep_going_this_spin = False # Force loop end
        # --- End Avalanche Loop ---

        # --- Post-Spin Processing ---
        remaining_spins -= 1
        total_fs_win += spin_win_this_fs

        if scatters_landed_this_fs_spin >= 2:
           additional_spins = calculate_retrigger_spins(scatters_landed_this_fs_spin)
           if additional_spins > 0:
               remaining_spins += additional_spins
               if verbose:
                   print(f"  Retrigger! +{additional_spins} spins (Scatters: {scatters_landed_this_fs_spin}). New Remaining: {remaining_spins}")

        ew_collected_session += ew_collected_this_fs_spin
        # Check for upgrades AFTER accumulating EWs from the completed spin
        if ew_collected_session >= config.FS_EW_COLLECTION_PER_UPGRADE:
            pending_upgrades = ew_collected_session // config.FS_EW_COLLECTION_PER_UPGRADE
            if pending_upgrades > 0:
                ew_collected_session -= pending_upgrades * config.FS_EW_COLLECTION_PER_UPGRADE
                spins_from_upgrade = pending_upgrades * config.FS_SPINS_PER_UPGRADE
                remaining_spins += spins_from_upgrade
                level_before = config.FS_BASE_MULTIPLIER_LEVELS[current_level_index]
                # Update index for the *next* spin
                current_level_index = min(current_level_index + pending_upgrades, len(config.FS_BASE_MULTIPLIER_LEVELS) - 1)
                level_after = config.FS_BASE_MULTIPLIER_LEVELS[current_level_index]
                if verbose:
                     print(f"  Multiplier Upgrade! {pending_upgrades} level(s). Base Mult x{level_before} -> x{level_after}. Spins Added: +{spins_from_upgrade}. New Remaining: {remaining_spins}. EW Counter now: {ew_collected_session}.")

        if verbose:
            print(f"--- End FS Spin {total_fs_spins_played} --- Spin Win: {spin_win_this_fs:.2f}, Total FS Win: {total_fs_win:.2f} ---")

    # --- End of Feature ---
    if verbose:
        print(f"{'='*15} Free Spins Finished {'='*15}")
        print(f"Total FS Win: {total_fs_win:.2f}")

    return total_fs_win

def run_simulation(num_spins: int, base_bet: float, run_id: str, verbose_spins: int = 0, 
                  verbose_fs_only: bool = False, return_stats: bool = False, enhanced_stats: bool = False):
    """
    Runs the simulation and logs results.
    
    Args:
        num_spins: Number of spins to simulate
        base_bet: Base bet amount
        run_id: Identifier for this run
        verbose_spins: Number of initial spins to display detailed output for (0 = none)
        verbose_fs_only: If True, show verbose output only during Free Spins features
        return_stats: If True, return a dictionary of key statistics (used for testing)
        enhanced_stats: If True, include enhanced statistics in the output
        
    Returns:
        Dict of statistics if return_stats is True, otherwise None
    """
    # Ensure log directory exists
    os.makedirs(LOG_DIR, exist_ok=True)
    summary_log_path = os.path.join(LOG_DIR, f"summary_{run_id}.txt")
    spins_log_path = os.path.join(LOG_DIR, f"spins_{run_id}.csv")

    # Create a GameState object for the Grid
    from simulator.core.state import GameState
    game_state = GameState()
    grid = Grid(game_state)
    total_win = 0.0
    total_scatters_seen = 0 # Total across all spins/avalanches
    spin_wins = [] # Store individual spin wins for stats
    win_distribution = defaultdict(int)
    fs_triggers = 0
    fs_total_win = 0.0 # Track total win from FS features
    
    # Enhanced statistics tracking
    top_wins = []  # Store top 10 wins with details
    win_ranges = [0, 0, 0, 0, 0, 0, 0]  # 0-1x, 1-5x, 5-10x, 10-50x, 50-100x, 100-500x, 500x+
    win_range_labels = ["0-1x", "1-5x", "5-10x", "10-50x", "50-100x", "100-500x", "500x+"]

    start_time = time.time()

    print(f"Starting simulation run '{run_id}' for {num_spins} spins...")
    print(f"Verbose Base Spins: {verbose_spins}")
    print(f"Verbose Free Spins Only: {verbose_fs_only}")
    print(f"Logging summary to: {summary_log_path}")
    print(f"Logging spin details to: {spins_log_path}")

    # Open CSV log file for writing spin details
    with open(spins_log_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        # Write header row - remove EW Collected column
        csv_writer.writerow(["SpinNumber", "TotalWin", "TotalScattersInSequence", "Hit"])

        for i in range(num_spins):
            # Determine verbosity for this specific spin/feature
            is_base_game_verbose = (i < verbose_spins) and not verbose_fs_only # Only verbose BG if -v is set AND --verbose-fs is NOT set
            is_free_spin_verbose = verbose_fs_only # FS is verbose if the flag is set

            spin_win, scatters_in_seq = run_base_game_spin(grid, base_bet, spin_index=i, verbose=is_base_game_verbose)

            current_spin_total_win = spin_win # Start with base game win
            total_scatters_seen += scatters_in_seq

            # Check for FS Trigger (GDD 4.8: 3+ scatters during a complete spin sequence)
            initial_free_spins = 0
            if scatters_in_seq >= 3:
                fs_triggers += 1
                # Calculate initial spins based on config
                initial_free_spins = config.FS_TRIGGER_SCATTERS.get(scatters_in_seq)
                if initial_free_spins is None: # Handle 5+ scatters
                    initial_free_spins = config.FS_TRIGGER_SCATTERS[4] + (scatters_in_seq - 4) * config.FS_TRIGGER_SCATTERS_EXTRA

                # --- Run Free Spins Feature --- (Call the new function)
                win_from_fs = run_free_spins_feature(grid, base_bet, initial_free_spins, trigger_spin_index=i, verbose=is_free_spin_verbose)
                fs_total_win += win_from_fs
                current_spin_total_win += win_from_fs # Add FS win to this spin's total win

            # Store the total win for this base game spin (including any FS win it triggered)
            spin_wins.append(current_spin_total_win)

            # Track information about the spin for detailed analysis
            spin_details = {
                'index': i + 1,
                'total_win': current_spin_total_win,
                'base_game_win': spin_win,
                'fs_win': current_spin_total_win - spin_win if scatters_in_seq >= 3 else 0,
                'scatters': scatters_in_seq,
                'triggered_fs': True if scatters_in_seq >= 3 else False,
                'win_multiplier': current_spin_total_win / base_bet if base_bet > 0 else 0
            }
            
            # Store detailed information about highest wins
            if len(top_wins) < 10:  # Keep track of top 10 wins
                top_wins.append(spin_details.copy())
                top_wins.sort(key=lambda x: x['total_win'], reverse=True)
            elif spin_details['total_win'] > top_wins[-1]['total_win']:
                top_wins[-1] = spin_details.copy()
                top_wins.sort(key=lambda x: x['total_win'], reverse=True)

            # Log base game spin result to CSV (FS details would need separate logging if desired)
            hit = 1 if current_spin_total_win > 0 else 0
            csv_writer.writerow([i + 1, f"{current_spin_total_win:.4f}", scatters_in_seq, hit])

            win_multiplier = round(current_spin_total_win / base_bet) if base_bet > 0 else 0
            win_distribution[win_multiplier] += 1
            
            # Add to win ranges (for analysis)
            if current_spin_total_win > 0:
                if current_spin_total_win <= base_bet:  # 0-1x
                    win_ranges[0] += 1
                elif current_spin_total_win <= base_bet * 5:  # 1-5x
                    win_ranges[1] += 1
                elif current_spin_total_win <= base_bet * 10:  # 5-10x
                    win_ranges[2] += 1
                elif current_spin_total_win <= base_bet * 50:  # 10-50x
                    win_ranges[3] += 1
                elif current_spin_total_win <= base_bet * 100:  # 50-100x
                    win_ranges[4] += 1
                elif current_spin_total_win <= base_bet * 500:  # 100-500x
                    win_ranges[5] += 1
                else:  # 500x+
                    win_ranges[6] += 1

            # Progress indicator (less frequent for large runs)
            if num_spins <= 1000 or (i + 1) % (num_spins // 100 if num_spins >= 100 else 1) == 0:
                progress = (i + 1) / num_spins * 100
                elapsed_time = time.time() - start_time
                print(f"Progress: {progress:.1f}% ({i+1}/{num_spins} spins). Elapsed time: {elapsed_time:.2f}s", end='\r') # Use carriage return

    print("\nSimulation calculations complete. Generating summary...") # Newline after progress indicator
    end_time = time.time()
    total_time = end_time - start_time

    # --- Calculate Final Stats --- (Adjust for FS win)
    total_win = sum(spin_wins) # Use accumulated spin_wins which includes FS results
    total_bet = num_spins * base_bet
    rtp = (total_win / total_bet) * 100 if total_bet > 0 else 0
    hit_count = len([w for w in spin_wins if w > 0])
    hit_frequency = (hit_count / num_spins) * 100 if num_spins > 0 else 0
    fs_trigger_freq_pct = (fs_triggers / num_spins) * 100 if num_spins > 0 else 0
    fs_trigger_freq_spins = num_spins / fs_triggers if fs_triggers > 0 else float('inf')
    spins_per_sec = num_spins / total_time if total_time > 0 else 0

    # --- Prepare Summary Output --- (Add FS Win Info)
    summary_lines = [
        f"--- Simulation Summary: {run_id} ---",
        f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Total Spins: {num_spins:,}",
        f"Base Bet: {base_bet:.2f}",
        f"Total Bet: {total_bet:,.2f}",
        f"Total Win (BG + FS): {total_win:,.2f}", # Clarify total win includes FS
        f"  Base Game Win: {(total_win - fs_total_win):,.2f}",
        f"  Free Spins Win: {fs_total_win:,.2f}",
        f"Return to Player (RTP): {rtp:.4f}%",
        "",
        f"Hit Count: {hit_count:,}",
        f"Hit Frequency: {hit_frequency:.2f}%",
        "",
        f"Total Scatters Seen (in sequences): {total_scatters_seen:,}",
        f"Free Spins Triggers (>=3 Scatters): {fs_triggers:,}",
        f"  FS Trigger Frequency: {fs_trigger_freq_pct:.4f}%",
        f"  FS Trigger Rate: ~1 in {fs_trigger_freq_spins:,.1f} spins",
        "",
        f"Simulation Time: {total_time:.2f} seconds",
        f"Spins per second: {spins_per_sec:.2f}",
        "",
    ]
    # Add top win multipliers to summary
    wins_sorted = sorted(win_distribution.items(), key=lambda item: item[1], reverse=True)
    # Add FS-specific stats
    summary_lines.append(f"  Avg Win per FS Trigger: {fs_total_win / fs_triggers if fs_triggers > 0 else 0:.2f}")

    # Always show Max Win statistic
    max_win_value = top_wins[0]['total_win'] if top_wins else 0
    max_win_mult = max_win_value / base_bet if base_bet > 0 else 0
    summary_lines.extend([
        "",
        f"Max Win: {max_win_value:,.2f} ({max_win_mult:.2f}x)"
    ])
    
    # Add enhanced statistics only if requested
    if enhanced_stats:
        summary_lines.extend([
            "",
            "--- Enhanced Statistics ---",
            "",
            f"Win Distribution by Ranges:",
        ])
        
        # Add win ranges statistics
        total_hits = sum(win_ranges)
        for i, (label, count) in enumerate(zip(win_range_labels, win_ranges)):
            if total_hits > 0:
                percentage = (count / total_hits) * 100
                summary_lines.append(f"  {label:<8}: {count:>8,} hits ({percentage:7.4f}% of hits)")
        
        # Add top wins section
        summary_lines.extend([
            "",
            f"Top 10 Wins:",
        ])
        
        # Add visualization for top wins
        if top_wins:
            max_win = top_wins[0]['total_win']
            bar_width = 40  # Max width of visualization bar
            
            for i, win in enumerate(top_wins):
                if max_win > 0:
                    bar_len = int((win['total_win'] / max_win) * bar_width)
                    bar = "█" * bar_len
                    mult = win['total_win'] / base_bet
                    fs_note = " (incl. FS)" if win['triggered_fs'] else ""
                    summary_lines.append(f"  {i+1:2d}. {win['total_win']:10,.2f} ({mult:7.2f}x){fs_note} | {bar}")
                else:
                    summary_lines.append(f"  {i+1:2d}. 0.00 (0.00x) |")

    # --- Write Summary to File & Print to Console ---
    summary_output = "\n".join(summary_lines)
    try:
        with open(summary_log_path, 'w') as f:
            f.write(summary_output)
    except IOError as e:
        print(f"Error writing summary log: {e}")

    print("\n" + summary_output)
    
    # Return statistics dictionary if requested (used for testing)
    if return_stats:
        return {
            'total_spins': num_spins,
            'total_bg_win': total_win - fs_total_win,
            'fs_total_win': fs_total_win,
            'total_win': total_win,
            'rtp': rtp,
            'hit_count': hit_count,
            'hit_frequency': hit_frequency,
            'total_scatters_seen': total_scatters_seen,
            'fs_triggers': fs_triggers,
            'fs_trigger_freq_pct': fs_trigger_freq_pct,
            'fs_trigger_freq_spins': fs_trigger_freq_spins,
            'win_distribution': win_distribution,
            'top_wins': top_wins,
            'win_ranges': win_ranges,
            'win_range_labels': win_range_labels,
            'max_win': top_wins[0]['total_win'] if top_wins else 0,
            'max_win_multiplier': top_wins[0]['total_win']/base_bet if top_wins else 0
        }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Esqueleto Explosivo 3 Clone Simulation")
    parser.add_argument("-n", "--num_spins", type=int, default=config.TOTAL_SIMULATION_SPINS,
                        help=f"Number of spins to simulate (default: {config.TOTAL_SIMULATION_SPINS})")
    # Verbose options - make mutually exclusive in logic if not via argparse
    parser.add_argument("-v", "--verbose", type=int, default=0,
                        help="Number of initial BASE GAME spins to run verbosely (ignored if -V is used)")
    parser.add_argument("-V", "--verbose-fs", action="store_true",
                        help="Run verbosely ONLY during Free Spins features.")
    parser.add_argument("-b", "--base_bet", type=float, default=config.BASE_BET,
                        help=f"Base bet amount (default: {config.BASE_BET})")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility (default: None)")
    parser.add_argument("--id", type=str, default=datetime.now().strftime('%Y%m%d_%H%M%S'),
                        help="Unique ID for this simulation run (default: timestamp)")
    parser.add_argument("--enhanced-stats", action="store_true",
                        help="Show enhanced statistics including max win, win distribution ranges, and visualizations")
    parser.add_argument("--stats-only", action="store_true",
                        help="Output only statistics to the console without running verbose mode")

    args = parser.parse_args()

    # If stats-only mode, set verbosity to 0
    if args.stats_only:
        verbose_bg_spins = 0
        verbose_fs = False
    else:
        # If verbose_fs is true, force verbose base game spins to 0
        verbose_bg_spins = 0 if args.verbose_fs else args.verbose
        verbose_fs = args.verbose_fs

    if args.seed is not None:
        print(f"Setting random seed: {args.seed}")
        random.seed(args.seed)

    # In stats-only mode, we automatically enable enhanced stats
    enhanced_stats = args.enhanced_stats or args.stats_only

    run_simulation(num_spins=args.num_spins,
                     base_bet=args.base_bet,
                     run_id=args.id,
                     verbose_spins=verbose_bg_spins, # Pass adjusted BG verbose count
                     verbose_fs_only=verbose_fs, # Pass the FS verbose flag
                     enhanced_stats=enhanced_stats) # Pass the enhanced stats flag
