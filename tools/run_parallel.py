#!/usr/bin/env python3
"""
Parallel-run shim for Esqueleto Explosivo 3 simulator.

Splits the total spin count across multiple processes and aggregates the results.
"""
import argparse
import datetime
import logging
import multiprocessing
import sys
import time

import tqdm

from simulator.main import run_simulation

# Setup logger for this module
logger = logging.getLogger(__name__)
# Basic configuration for the logger.
# This will be configured more specifically in main() if the script is run directly.
if not logger.hasHandlers():
    _handler = logging.StreamHandler(sys.stdout)
    _formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    _handler.setFormatter(_formatter)
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)


# Worker returns only stats dict with worker_id
def _run_worker(
    spins: int, base_bet: float, seed: int, idx: int, log_level_str: str
) -> dict:
    # Configure logging for the worker process
    # Each worker should have its own logger configuration to avoid issues with shared handlers
    # or to allow per-worker log levels if needed.
    # For now, inherit root logger config or use a basic one.
    # The main simulator.main logger is already configured, this is for run_parallel's own logs.

    # If run_simulation itself configures logging, ensure it's reentrant or handled.
    # simulator.main.logger is specific to that module.
    # This _run_worker is in a different process.

    # Minimal config for this worker's context if its logs are needed:
    # logging.basicConfig(level=log_level_str.upper(), format='%(asctime)s - WORKER %(levelname)s - %(message)s', force=True)

    stats = run_simulation(
        num_spins=spins,
        base_bet=base_bet,
        run_id=f"worker_{idx}",
        return_stats=True,
        verbose_spins=0,  # Workers should not be verbose to avoid clutter
        seed=seed,
        calc_roe_flag=False,  # ROE is typically calculated on aggregated results, not per worker
    )
    stats["worker_id"] = idx
    return stats


def _run_worker_star(args_dict):  # Modified to accept a dictionary
    return _run_worker(**args_dict)


def main():
    parser = argparse.ArgumentParser(description="Parallel batch-run for the simulator")
    parser.add_argument(
        "--spins", type=int, required=True, help="Total number of spins to distribute"
    )
    parser.add_argument("--bet", type=float, default=1.0, help="Base bet amount")
    parser.add_argument(
        "--cores",
        type=int,
        default=multiprocessing.cpu_count(),
        help="Number of parallel workers",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Base seed for RNG streams"
    )
    parser.add_argument(
        "--log",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level for this script.",
    )

    args = parser.parse_args()

    # --- Setup Logging Handler and Level for this script ---
    log_level_numeric = getattr(logging, args.log.upper(), None)
    if not isinstance(log_level_numeric, int):
        raise ValueError(f"Invalid log level: {args.log}")

    # Clear existing handlers for this specific logger instance (run_parallel's logger)
    if logger.hasHandlers():
        logger.handlers.clear()

    # Add a new handler for this script's logger
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(log_level_numeric)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logger.setLevel(log_level_numeric)

    # Also configure the root logger if desired, or rely on simulator.main's config for its own logs.
    # For run_simulation logs to appear based on its internal --log like flag,
    # it would need to be passed or simulator.main.py needs to be import-safe with its logging.
    # The run_simulation function itself now handles its logger based on its args or internal defaults.

    total_spins = args.spins
    base_bet = args.bet
    cores = args.cores
    base_seed = args.seed

    start_time = time.time()

    spins_per_core, remainder = divmod(total_spins, cores)
    # Pass log_level_str to worker if it needs to configure its own logging.
    tasks = [
        {
            "spins": spins_per_core + (1 if i < remainder else 0),
            "base_bet": base_bet,
            "seed": base_seed + i,
            "idx": i,
            "log_level_str": args.log,  # Pass the log level string
        }
        for i in range(cores)
    ]

    results = []
    # tqdm for the pool submission, ensure it goes to stderr
    with multiprocessing.Pool(cores) as pool:
        for stats in tqdm.tqdm(
            pool.imap_unordered(_run_worker_star, tasks),
            total=len(tasks),
            desc="Workers",
            unit="worker",
            file=sys.stderr,
        ):
            results.append(stats)

    # logger.info per-worker summary
    logger.info("--- Worker Summaries ---")
    for st in sorted(results, key=lambda s: s["worker_id"]):
        wid = st["worker_id"]
        # Ensure stats keys exist or provide defaults
        rtp_val = st.get("rtp", 0)
        hit_freq_val = st.get(
            "hit_frequency", 0
        )  # Changed from 'hit_freq' to 'hit_frequency' based on run_simulation stats dict
        logger.info(
            f"Worker {wid}: Spins={st.get('total_spins',0)}, TotalWin={st.get('total_win', 0):.2f}, RTP={rtp_val:.4f}%, HitFreq={hit_freq_val:.2f}%"
        )

    # Aggregate results
    total_spins_run = sum(r.get("total_spins", 0) for r in results)
    total_win = sum(r.get("total_win", 0) for r in results)
    # aggregated_total_staked = sum(r.get('total_staked', 0) for r in results) # More robust if base_bet varies
    aggregated_total_staked = (
        total_spins_run * base_bet
    )  # Assuming base_bet is constant for all workers

    rtp = (
        (total_win / aggregated_total_staked) * 100
        if aggregated_total_staked > 0
        else 0
    )
    total_hits = sum(r.get("hit_count", 0) for r in results)
    hit_freq_agg = (
        (total_hits / total_spins_run) * 100 if total_spins_run > 0 else 0
    )  # Renamed to avoid clash

    # New aggregations from stats dictionary
    total_base_game_win_agg = sum(r.get("total_bg_win", 0) for r in results)
    total_free_spins_win_agg = sum(
        r.get("fs_total_win", 0) for r in results
    )  # Key changed from 'total_fs_win'
    total_scatters_seen_agg = sum(r.get("total_scatters_seen", 0) for r in results)
    total_fs_triggers_agg = sum(r.get("fs_triggers", 0) for r in results)

    max_overall_abs_win = 0
    max_overall_win_x = 0
    if results:  # Ensure results is not empty
        # Filter out Nones if any worker failed and returned None (or handle more gracefully)
        valid_results = [r for r in results if r is not None and isinstance(r, dict)]
        if valid_results:
            max_overall_abs_win = max(
                r.get("max_win", 0)
                for r in valid_results
                if r.get("max_win") is not None
            )
            max_overall_win_x = max(
                r.get("max_win_multiplier", 0)
                for r in valid_results
                if r.get("max_win_multiplier") is not None
            )

    end_time = time.time()
    simulation_duration = end_time - start_time

    # Derived stats for new output
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    fs_trigger_freq_agg_val = (
        (total_fs_triggers_agg / total_spins_run) * 100 if total_spins_run > 0 else 0
    )  # Renamed

    fs_trigger_rate_str = "N/A"
    if total_fs_triggers_agg > 0:
        fs_trigger_rate_val = total_spins_run / total_fs_triggers_agg
        fs_trigger_rate_str = f"~1 in {fs_trigger_rate_val:.1f} spins"

    spins_per_second_val = (
        total_spins_run / simulation_duration if simulation_duration > 0 else 0
    )
    avg_win_per_fs_trigger_val = (
        total_free_spins_win_agg / total_fs_triggers_agg
        if total_fs_triggers_agg > 0
        else 0
    )

    # --- Updated Aggregated Summary (using logger.info) ---
    logger.info("\\n--- Aggregated Summary for all workers ---")
    logger.info(f"Timestamp: {timestamp}")
    logger.info(f"Total Spins: {total_spins_run:,}")
    logger.info(f"Base Bet: {base_bet:.2f}")
    logger.info(f"Total Bet: {aggregated_total_staked:,.2f}")
    logger.info(f"Total Win (BG + FS): {total_win:,.2f}")
    logger.info(f"  Base Game Win: {total_base_game_win_agg:,.2f}")
    logger.info(f"  Free Spins Win: {total_free_spins_win_agg:,.2f}")
    logger.info(f"Return to Player (RTP): {rtp:.4f}%")
    # ROE is not calculated by run_parallel.py directly.
    # logger.info("Median ROE: Infinite")
    # logger.info("Average ROE: Infinite")
    logger.info(f"\\nHit Count: {total_hits:,}")
    logger.info(f"Hit Frequency: {hit_freq_agg:.2f}%")  # Use renamed variable
    logger.info(f"\\nTotal Scatters Seen (in sequences): {total_scatters_seen_agg:,}")
    logger.info(f"Free Spins Triggers (>=3 Scatters): {total_fs_triggers_agg:,}")
    logger.info(
        f"  FS Trigger Frequency: {fs_trigger_freq_agg_val:.4f}%"
    )  # Use renamed variable
    logger.info(f"  FS Trigger Rate: {fs_trigger_rate_str}")
    logger.info(f"\\nSimulation Time: {simulation_duration:.2f} seconds")
    logger.info(f"Spins per second: {spins_per_second_val:.2f}")
    logger.info(f"\\n  Avg Win per FS Trigger: {avg_win_per_fs_trigger_val:,.2f}")
    logger.info(f"\\nMax Win: {max_overall_abs_win:,.2f} ({max_overall_win_x:,.2f}x)")


if __name__ == "__main__":
    main()
