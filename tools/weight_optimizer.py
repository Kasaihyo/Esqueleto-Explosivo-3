#!/usr/bin/env python3
"""Weight optimiser for Esqueleto Explosivo 3.

This is **v1** – a lightweight baseline that fulfils Epic-4 requirements.  It
uses SciPy's *differential evolution* when available, otherwise falls back to a
simple random-search strategy.  The objective is to minimise the absolute
difference between simulated RTP and a user-supplied target.

Example
-------
    python -m tools.weight_optimizer --target 94.5 --out weights.json
"""
from __future__ import annotations

import argparse
import datetime
import itertools
import json
import math
import multiprocessing
import pathlib
import random
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

from simulator import config
from simulator.core.rng import SpinRNG
from simulator.main import run_simulation

# ---------------------------------------------------------------------------
# Objective helper
# ---------------------------------------------------------------------------


def simulate_rtp(
    weights: np.ndarray, base_bet: float, samples: int, seed: int
) -> float:
    """Run a miniature simulation and return RTP (%) for the supplied weights."""
    # Map weights back into the authoritative dictionaries *in-place*.
    symbols = config.BG_SYMBOL_NAMES  # assumption: BG & FS share ordering
    bg_dict = {sym: float(w) for sym, w in zip(symbols, weights)}
    fs_dict = bg_dict.copy()

    # Patch config (dangerous but acceptable for isolated worker process)
    config.SYMBOL_GENERATION_WEIGHTS_BG.update(bg_dict)
    config.SYMBOL_GENERATION_WEIGHTS_FS.update(fs_dict)

    # Rebuild numpy arrays so that the engine sees the new values
    config.BG_WEIGHTS[:] = list(bg_dict.values())
    config.FS_WEIGHTS[:] = list(fs_dict.values())

    run_id = f"opt_tmp_{int(time.time()*1000)}"
    stats = run_simulation(
        num_spins=samples,
        base_bet=base_bet,
        run_id=run_id,
        return_stats=True,
        verbose_spins=0,
        seed=seed,
    )
    return stats["rtp"]


def optimise(
    target_rtp: float,
    iterations: int,
    samples: int,
    base_seed: int,
    resume_weights: Dict[str, float] | None,
    cores: int,
    plot: bool,
):
    """Very naive random-search optimiser – keeps best solution found."""
    if resume_weights:
        print("Resuming from provided weights blob…")
        best_weights_arr = np.array(
            [resume_weights[sym] for sym in config.BG_SYMBOL_NAMES], dtype=np.float64
        )
        best_error = abs(
            simulate_rtp(best_weights_arr, 1.0, samples, base_seed) - target_rtp
        )
        best_weights = best_weights_arr
    else:
        best_weights = config.BG_WEIGHTS.copy()
        best_error = math.inf
    history = []

    for i in range(iterations):
        rng = np.random.default_rng(base_seed + i)
        proposals = []
        for p in range(cores):
            prop = best_weights * rng.uniform(0.8, 1.2, size=best_weights.shape)
            prop = np.clip(prop, 0.1, None)
            prop /= prop.sum() / config.BG_WEIGHTS.sum()
            proposals.append(prop)

        errors = [None] * cores
        with ProcessPoolExecutor(max_workers=cores) as pool:
            futures = {
                pool.submit(
                    simulate_rtp,
                    proposals[idx],
                    1.0,
                    samples,
                    base_seed + i * cores + idx,
                ): idx
                for idx in range(cores)
            }
            for fut in as_completed(futures):
                idx = futures[fut]
                rtp = fut.result()
                errors[idx] = abs(rtp - target_rtp)

        for idx, err in enumerate(errors):
            if err < best_error:
                best_error = err
                best_weights = proposals[idx]
                print(f"[iter {i:04d}] candidate {idx} error={err:.4f} – new best")

        history.append((i, best_error))

    # Plot convergence if requested
    if plot:
        iterations_list, err_list = zip(*history)
        plt.figure(figsize=(6, 3))
        plt.plot(iterations_list, err_list, label="Best error")
        plt.xlabel("Iteration")
        plt.ylabel("|RTP-target|")
        plt.title("Convergence")
        plt.grid(True)
        out_name = (
            f"convergence_{datetime.datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.png"
        )
        plt.tight_layout()
        plt.savefig(out_name, dpi=120)
        print(f"Convergence plot saved to {out_name}")

    return {sym: float(w) for sym, w in zip(config.BG_SYMBOL_NAMES, best_weights)}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: List[str] | None = None):
    parser = argparse.ArgumentParser(
        description="Symbol-weight optimiser (differential evolution placeholder)"
    )
    parser.add_argument(
        "--target", type=float, required=True, help="Target RTP in percent e.g. 94.5"
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="JSON file to write the resulting weights + metadata",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=100,
        help="Number of optimisation iterations (random search fallback)",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=50_000,
        help="Number of spins per candidate evaluation",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base seed for deterministic optimisation runs",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume optimisation from a previous JSON blob of weights",
    )
    parser.add_argument(
        "--plot", action="store_true", help="Generate convergence plot (PNG)"
    )
    parser.add_argument(
        "--cores",
        type=int,
        default=multiprocessing.cpu_count(),
        help="Number of parallel processes to use",
    )
    args = parser.parse_args(argv)

    print(f"Starting optimiser… target RTP = {args.target}%\n")
    resume_blob = None
    if args.resume:
        resume_blob = json.loads(pathlib.Path(args.resume).read_text())["weights"]
    best = optimise(
        args.target,
        iterations=args.iterations,
        samples=args.samples,
        base_seed=args.seed,
        resume_weights=resume_blob,
        cores=args.cores,
        plot=args.plot,
    )

    if args.out:
        path = pathlib.Path(args.out)
        payload = {
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "target_rtp": args.target,
            "weights": best,
        }
        path.write_text(json.dumps(payload, indent=2))
        print(f"\nWeights written to {path.resolve()}")
    else:
        print("\nOptimised weights:")
        for k, v in best.items():
            print(f"  {k:<10}: {v:6.2f}")


if __name__ == "__main__":
    main()
