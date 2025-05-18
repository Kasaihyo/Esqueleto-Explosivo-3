#!/usr/bin/env python
"""Auto-optimized simulator launcher for Esqueleto Explosivo 3.

This script automatically detects hardware capabilities and runs the simulator
with optimal settings, with special optimizations for Apple Silicon ARM processors.
It provides a unified interface for all simulator features including RNG handling,
ROE calculation, and hardware-specific optimizations.

Usage:
    python run.py [options]

Basic Examples:
    python run.py                  # Run with auto-detected optimal settings
    python run.py --spins 100000   # Run 100,000 spins
    python run.py --bet 2.0        # Set bet amount to 2.0
    python run.py --seed 12345     # Use specific random seed

RNG Handling Examples:
    python run.py --identical-sequence          # Force identical results (default)
    python run.py --no-identical-sequence       # Allow parallel RNG (faster)
    python run.py --no-sequential-rng           # Maximum performance RNG

Hardware Profiles:
    python run.py --profile m1                  # Optimize for M1 chip
    python run.py --profile m1pro               # Optimize for M1 Pro chip
    python run.py --profile 11core18gb          # Optimize for 11-core Apple Silicon

For complete documentation, see docs/commands.md
"""

import sys
import subprocess
import argparse

# Check for Python 3.8+
if sys.version_info < (3, 8):
    print("Error: Python 3.8 or later is required.")
    sys.exit(1)

# Ensure required dependencies are installed
def _install_deps():
    print("Installing required dependencies...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-e", "."], check=True)

for _dep in ("numba", "joblib"):
    try:
        __import__(_dep)
    except ImportError:
        _install_deps()
        break
# import datetime # Redundant: from datetime import datetime is used
# import logging # F401 unused
import multiprocessing
import platform
# import random # F401 unused
import subprocess
# import time # F401 unused
# import warnings # F401 unused
from datetime import datetime

import psutil

# Check if GPU is available by trying to import CUDA modules
CUDA_AVAILABLE = False
try:
    import numba.cuda

    CUDA_AVAILABLE = numba.cuda.is_available()
except ImportError:
    pass

# Check for Apple Silicon
IS_APPLE_SILICON = platform.system() == "Darwin" and platform.machine() == "arm64"

# Get available CPU cores and memory
NUM_CORES = multiprocessing.cpu_count()
AVAILABLE_MEMORY_GB = psutil.virtual_memory().total / (1024**3)


def get_hardware_info():
    """Get hardware information as a dictionary"""
    return {
        "cpu_count": NUM_CORES,
        "memory_gb": AVAILABLE_MEMORY_GB,
        "is_apple_silicon": IS_APPLE_SILICON,
        "has_cuda": CUDA_AVAILABLE,
        "platform": f"{platform.system()} {platform.release()}",
        "processor": platform.processor() or ("arm" if IS_APPLE_SILICON else "unknown"),
    }


def get_optimal_settings(profile=None):
    """
    Determine optimal settings based on hardware and selected profile.

    Args:
        profile: Optional hardware profile to optimize for ('m1', 'm1pro', 'm2', etc.)

    Returns:
        Dictionary with optimal settings
    """
    settings = {}
    hw_info = get_hardware_info()

    # Default values
    settings["num_spins"] = 50000  # Default spins
    settings["batch_size"] = 1000  # Default batch size
    settings["cores"] = min(NUM_CORES, 8)  # Use at most 8 cores by default
    settings["no_gpu"] = not (
        CUDA_AVAILABLE or IS_APPLE_SILICON
    )  # Use GPU if available
    settings["enhanced_stats"] = False
    settings["enable_jit"] = True

    # Memory-based adjustments
    if AVAILABLE_MEMORY_GB >= 32:
        settings["num_spins"] = 1000000
        settings["batch_size"] = 10000
    elif AVAILABLE_MEMORY_GB >= 16:
        settings["num_spins"] = 500000
        settings["batch_size"] = 5000
    elif AVAILABLE_MEMORY_GB >= 8:
        settings["num_spins"] = 100000
        settings["batch_size"] = 2000
    else:
        settings["num_spins"] = 50000
        settings["batch_size"] = 1000

    # CPU adjustments
    if NUM_CORES >= 16:
        settings["cores"] = NUM_CORES - 4  # Leave some cores for OS
    elif NUM_CORES >= 8:
        settings["cores"] = NUM_CORES - 2
    else:
        settings["cores"] = max(1, NUM_CORES - 1)  # At least 1 core

    # Apply specific optimizations for Apple Silicon
    if IS_APPLE_SILICON:
        # Apple Silicon optimization - adjusted batch size based on memory
        settings["batch_size"] = min(int(AVAILABLE_MEMORY_GB * 300), 10000)

        # For 11-core Apple Silicon (M1 Pro/Max or M2 Pro/Max)
        if NUM_CORES == 11:
            settings["cores"] = 8  # Use 8 cores (likely 8 performance cores)
            settings["batch_size"] = 6000  # Optimized for ~18GB RAM
            settings["enable_jit"] = True

        # Optimize based on memory available
        if AVAILABLE_MEMORY_GB >= 16 and AVAILABLE_MEMORY_GB < 24:
            # Specific optimization for ~18GB RAM devices (M1 Pro, M2 Pro)
            settings["batch_size"] = 6000
            settings["num_spins"] = 750000
    elif CUDA_AVAILABLE:
        settings["batch_size"] = 10000  # CUDA GPU handles large batches well

    # Profile-specific optimizations
    if profile:
        # Predefined profiles for common hardware configurations
        profiles = {
            "m1": {"cores": min(7, NUM_CORES), "batch_size": 3000, "enable_jit": True},
            "m1pro": {
                "cores": min(8, NUM_CORES),
                "batch_size": 6000,
                "enable_jit": True,
            },
            "m1max": {
                "cores": min(8, NUM_CORES),
                "batch_size": 8000,
                "enable_jit": True,
            },
            "m2": {"cores": min(7, NUM_CORES), "batch_size": 4000, "enable_jit": True},
            "m2pro": {
                "cores": min(8, NUM_CORES),
                "batch_size": 6000,
                "enable_jit": True,
            },
            "m3": {
                "cores": min(NUM_CORES - 2, NUM_CORES),
                "batch_size": 8000,
                "enable_jit": True,
            },
        }

        if profile in profiles:
            for key, value in profiles[profile].items():
                settings[key] = value
        elif profile == "11core18gb":
            # Exact match for the detected hardware (11 cores, 18GB RAM)
            settings["cores"] = 10  # Use 10 cores for maximum throughput
            settings["batch_size"] = 10000  # Larger batch size for better throughput
            settings["num_spins"] = 1_000_000
            settings["enable_jit"] = True
            # Add Apple Silicon specific flags
            settings["apple_silicon_turbo"] = True
            settings[
                "threading_model"
            ] = "openmp"  # Use OpenMP for better threading performance

    return settings


def print_hardware_info():
    """Print detected hardware information"""
    hw_info = get_hardware_info()
    print("\n=== Hardware Detection ===")
    print(f"CPU: {hw_info['processor']}")
    print(f"Cores: {hw_info['cpu_count']}")
    print(f"Memory: {hw_info['memory_gb']:.1f} GB")
    print(f"Platform: {hw_info['platform']}")
    print(f"Apple Silicon: {'Yes' if hw_info['is_apple_silicon'] else 'No'}")
    print(f"CUDA Available: {'Yes' if hw_info['has_cuda'] else 'No'}")


def main():
    parser = argparse.ArgumentParser(
        description="Auto-optimized simulator for Esqueleto Explosivo 3"
    )
    parser.add_argument(
        "spins_positional",
        type=int,
        nargs="?",
        help="Positional number of spins (alias for --spins)",
    )
    parser.add_argument(
        "--spins",
        type=int,
        default=None,
        help="Number of spins (default: auto-detect based on hardware)",
    )
    parser.add_argument(
        "--plots",
        action="store_true",
        help="Generate visualization plots (requires matplotlib)",
    )
    parser.add_argument(
        "--profile",
        type=str,
        choices=["m1", "m1pro", "m1max", "m2", "m2pro", "m3", "11core18gb"],
        help="Optimization profile for specific hardware",
    )
    parser.add_argument(
        "--debug", action="store_true", help="Run with enhanced debugging output"
    )
    parser.add_argument(
        "--bet", type=float, default=1.0, help="Base bet amount (default: 1.0)"
    )
    parser.add_argument(
        "--roe",
        action="store_true",
        dest="calculate_roe",
        default=True,
        help="Calculate ROE statistics (default: enabled)",
    )
    parser.add_argument(
        "--no-roe",
        action="store_false",
        dest="calculate_roe",
        help="Disable ROE calculation",
    )
    parser.add_argument(
        "--roe-use-main-data",
        action="store_true",
        default=True,
        help="Use main simulation data for ROE calculation (faster)",
    )
    parser.add_argument(
        "--roe-separate-sims",
        action="store_false",
        dest="roe_use_main_data",
        help="Run separate simulations for ROE calculation (slower)",
    )
    parser.add_argument(
        "--roe-num-sims",
        type=int,
        default=1000,
        help="Number of ROE simulations to run if not using main data (default: 1000)",
    )
    # Add RNG handling flags
    parser.add_argument(
        "--identical-sequence",
        action="store_true",
        default=True,
        help="Force identical results as non-optimized version when using same seed (default: True)",
    )
    parser.add_argument(
        "--no-identical-sequence",
        dest="identical_sequence",
        action="store_false",
        help="Allow optimized version to use parallel RNG (faster but different results from non-optimized)",
    )
    parser.add_argument(
        "--sequential-rng",
        action="store_true",
        default=True,
        help="Simulate sequential RNG behavior in parallel execution (default: True)",
    )
    parser.add_argument(
        "--no-sequential-rng",
        dest="sequential_rng",
        action="store_false",
        help="Use fully parallel RNG for maximum performance (may affect RTP)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Set a specific random seed for reproducible results",
    )
    args = parser.parse_args()
    # Handle positional spins argument (alias for --spins)
    if args.spins_positional is not None:
        args.spins = args.spins_positional

    # Get optimal settings based on hardware and profile
    settings = get_optimal_settings(args.profile)

    # If 11-core device with ~18GB RAM is detected, suggest the specific profile
    hw_info = get_hardware_info()
    if (
        hw_info["cpu_count"] == 11
        and 16 <= hw_info["memory_gb"] < 24
        and not args.profile
    ):
        print("\n⚠️  Detected 11-core Apple Silicon with ~18GB RAM")
        print("    For best performance, consider using: --profile 11core18gb")

        # Automatically apply the optimized profile
        settings = get_optimal_settings("11core18gb")
        print("    ✓ Automatically applying optimized settings for this hardware")

    # Override with user-provided spins if any
    if args.spins is not None:
        settings["num_spins"] = args.spins

    # Create a unique run ID with timestamp
    run_id = f"auto_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Print hardware info
    print_hardware_info()

    # Create command with optimal parameters
    cmd = [
        "python",
        "-m",
        "simulator.main",
        "-n",
        str(settings["num_spins"]),
        "--batch-size",
        str(settings["batch_size"]),
        "--cores",
        str(settings["cores"]),
        "--stats-only",
        "--id",
        run_id,
        "--bet",
        str(args.bet),
    ]

    # Add additional flags
    if settings["no_gpu"]:
        cmd.append("--no-gpu")

    if args.plots:
        cmd.append("--create-plots")
        cmd.append("--enhanced-stats")

    if args.debug:
        cmd.append("--verbose")
        cmd.append("--enhanced-stats")

    if not settings["enable_jit"]:
        cmd.append("--no-jit")

    # Add Apple Silicon specific optimizations if available
    if settings.get("apple_silicon_turbo", False):
        cmd.append("--turbo")

    if settings.get("threading_model"):
        cmd.append("--threading")
        cmd.append(settings["threading_model"])

    # Add ROE calculation flags
    if not args.calculate_roe:
        cmd.append("--no-roe")
    elif not args.roe_use_main_data:
        cmd.append("--roe-separate-sims")

    if args.roe_num_sims != 1000:
        cmd.append("--roe-num-sims")
        cmd.append(str(args.roe_num_sims))

    # Add RNG handling flags
    if not args.identical_sequence:
        cmd.append("--no-identical-sequence")

    if not args.sequential_rng:
        cmd.append("--no-sequential-rng")

    if args.seed is not None:
        cmd.append("--seed")
        cmd.append(str(args.seed))

    # Print command details
    print("\n=== Running with Optimal Settings ===")
    print(f"Spins: {settings['num_spins']:,}")
    print(f"Batch Size: {settings['batch_size']}")
    print(f"Cores: {settings['cores']}")
    print(f"GPU Acceleration: {'Disabled' if settings['no_gpu'] else 'Enabled'}")
    print(f"JIT Compilation: {'Enabled' if settings['enable_jit'] else 'Disabled'}")
    print(f"Visualization: {'Enabled' if args.plots else 'Disabled'}")
    print(
        f"Turbo Mode: {'Enabled' if settings.get('apple_silicon_turbo', False) else 'Disabled'}"
    )
    print(f"Threading Model: {settings.get('threading_model', 'default')}")

    # Print ROE settings
    roe_status = "Disabled" if not args.calculate_roe else "Enabled"
    roe_method = (
        "Using main data (fast)"
        if args.roe_use_main_data
        else f"Separate sims ({args.roe_num_sims})"
    )
    print(f"ROE Calculation: {roe_status} - {roe_method if args.calculate_roe else ''}")

    # Print RNG settings
    print(
        f"RNG Mode: {'Force identical with non-optimized' if args.identical_sequence else 'Optimized parallel'}"
    )
    if not args.identical_sequence:
        print(
            f"Sequential RNG: {'Simulated (consistent RTP)' if args.sequential_rng else 'Disabled (max performance)'}"
        )
    if args.seed is not None:
        print(f"Random Seed: {args.seed}")

    print(f"Run ID: {run_id}")
    print("\nStarting simulation...")

    # Run the command
    subprocess.run(cmd)

    print(
        f"\nSimulation complete! Results saved to simulation_results/summary_{run_id}.txt"
    )
    if args.plots:
        print(f"Plots saved to simulation_results/plots/{run_id}_*.png")


if __name__ == "__main__":
    main()
