#!/usr/bin/env python3
"""
Compare Non-optimized and Optimized (stub) simulator runs.

This script runs 100,000 spins for both versions with the same seed
and prints their statistics for comparison.
"""
import random

from simulator.main import run_simulation


def main():
    num_spins = 100_000
    base_bet = 1.0
    seed = 42

    # Run non-optimized version (suppress verbose output)
    import io
    import sys
    from contextlib import redirect_stdout

    buf_main = io.StringIO()
    random.seed(seed)
    with redirect_stdout(buf_main):
        stats_main = run_simulation(
            num_spins, base_bet, "main", return_stats=True, seed=seed
        )

    # Run optimized version (stub: same code)
    buf_opt = io.StringIO()
    random.seed(seed)
    with redirect_stdout(buf_opt):
        stats_opt = run_simulation(
            num_spins, base_bet, "opt", return_stats=True, seed=seed
        )

    print("\n--- Non-Optimized Stats ---")
    keys = [
        "total_win",
        "rtp",
        "hit_count",
        "hit_frequency",
        "fs_triggers",
        "total_scatters_seen",
    ]
    for k in keys:
        print(f"{k}: {stats_main.get(k)}")

    print("\n--- Optimized Stats (simulated) ---")
    for k in keys:
        print(f"{k}: {stats_opt.get(k)}")


if __name__ == "__main__":
    main()
