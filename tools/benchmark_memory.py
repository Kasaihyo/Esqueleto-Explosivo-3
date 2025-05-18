#!/usr/bin/env python3
"""
Memory profiler benchmark for the Esqueleto Explosivo 3 simulator.

Ensures that a long simulation run (10 million spins) stays within
the 1 GB memory limit.

To run:
  pip install memory_profiler (if not already installed)
  python -m tools.benchmark_memory
"""
import os
import sys
import time

# Adjust path to import simulator modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from simulator import config  # To potentially set JIT, etc.
from simulator.main import run_simulation

# Conditional import for memory_profiler
try:
    from memory_profiler import memory_usage
except ImportError:
    print(
        "Error: memory_profiler package not found. Please install it: pip install memory_profiler"
    )
    sys.exit(1)

# Benchmark parameters
NUM_SPINS_FOR_BENCHMARK = 10_000_000
MEMORY_LIMIT_MB = 1024  # 1 GB


def simulation_task_for_memory_check():
    """
    The actual task that will be memory profiled.
    Runs the simulation with minimal overhead.
    """
    print(f"Starting memory benchmark: Simulating {NUM_SPINS_FOR_BENCHMARK:,} spins...")
    # Ensure JIT is off for a consistent baseline, unless specifically testing JIT memory.
    # For this benchmark, we primarily care about Python object accumulation.
    original_enable_jit = config.ENABLE_JIT
    config.ENABLE_JIT = False

    start_time = time.time()

    # Run simulation with minimal features enabled to focus on core memory usage
    stats = run_simulation(
        num_spins=NUM_SPINS_FOR_BENCHMARK,
        base_bet=1.0,
        run_id="mem_bench",
        return_stats=True,  # Return stats to ensure full execution
        verbose_spins=0,  # No verbose spin logging
        verbose_fs_only=False,  # No verbose FS logging
        calc_roe_flag=False,  # Skip ROE calculation
        enhanced_stats=False,  # Skip enhanced stats
        seed=12345,  # Use a fixed seed for reproducibility of the run itself
    )

    end_time = time.time()
    duration = end_time - start_time

    # Restore original JIT setting
    config.ENABLE_JIT = original_enable_jit

    print(f"Simulation part of benchmark finished in {duration:.2f} seconds.")
    if stats.get("rtp"):
        print(
            f"Simulation produced RTP: {stats['rtp']:.4f}% (for validation of run completion)"
        )
    else:
        print("Warning: Simulation did not return RTP. Stats:", stats)


def run_memory_benchmark():
    """
    Runs the simulation task under memory_profiler and checks peak usage.
    """
    print("Preparing to run simulation task for memory profiling...")

    # Using memory_usage to get the peak memory of the target function.
    # We pass the function and its arguments separately.
    # `max_usage=True` makes it return only the peak memory.
    # `retval=True` would make it return (peak_mem, return_value_of_function)
    # For this, we only need peak_mem.

    # Run the simulation_task_for_memory_check once to allow for any Numba compilation
    # or one-time setup if JIT were enabled, so it doesn't count towards profiled memory.
    # However, we explicitly disable JIT in the task for this benchmark.
    print("Performing a warm-up run (not profiled)...")
    simulation_task_for_memory_check()
    print("Warm-up run complete.")

    print("\nStarting profiled run...")
    # The memory_usage function expects a tuple: (callable, args, kwargs)
    # Our target function `simulation_task_for_memory_check` takes no arguments.
    mem_profile_data = memory_usage(
        (simulation_task_for_memory_check, (), {}),
        interval=0.5,  # Check memory every 0.5 seconds
        timeout=None,  # No timeout for the profiler itself (simulation has its own implicit timeout)
        max_usage=True,  # Get only the peak memory usage
    )

    # mem_profile_data will be a single float if max_usage=True (peak memory in MiB)
    peak_memory_mb = mem_profile_data

    print(f"\n--- Memory Benchmark Results ---")
    print(f"Peak memory usage: {peak_memory_mb:.2f} MiB")
    print(f"Memory limit: {MEMORY_LIMIT_MB} MiB")

    if peak_memory_mb < MEMORY_LIMIT_MB:
        print(f"SUCCESS: Peak memory usage is within the {MEMORY_LIMIT_MB} MiB limit.")
        return True
    else:
        print(
            f"FAILURE: Peak memory usage {peak_memory_mb:.2f} MiB exceeds the {MEMORY_LIMIT_MB} MiB limit!"
        )
        # Optionally, exit with an error code for CI systems
        # sys.exit(1)
        return False


if __name__ == "__main__":
    print("=== Esqueleto Explosivo 3 - Memory Benchmark ===")
    benchmark_passed = run_memory_benchmark()
    if not benchmark_passed:
        sys.exit(1)  # Exit with error if benchmark failed
