#!/usr/bin/env python
"""
Auto-optimized simulator launcher for Esqueleto Explosivo 3.
This script automatically detects hardware capabilities and runs the simulator
with optimal settings.

Usage:
    python run.py [spins]

Example:
    python run.py           # Run with auto-detected optimal settings
    python run.py 100000    # Run 100,000 spins
"""

import os
import sys
import platform
import multiprocessing
import psutil
import argparse
import subprocess
from datetime import datetime

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
AVAILABLE_MEMORY_GB = psutil.virtual_memory().total / (1024 ** 3)

def get_optimal_settings():
    """Determine optimal settings based on hardware"""
    settings = {}
    
    # Default values
    settings["num_spins"] = 50000  # Default spins
    settings["batch_size"] = 1000  # Default batch size
    settings["cores"] = min(NUM_CORES, 8)  # Use at most 8 cores by default
    settings["no_gpu"] = not (CUDA_AVAILABLE or IS_APPLE_SILICON)  # Use GPU if available
    
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
        settings["cores"] = 12  # Leave some cores for OS
    elif NUM_CORES >= 8:
        settings["cores"] = 6
    else:
        settings["cores"] = max(1, NUM_CORES - 1)  # At least 1 core
    
    # GPU/Apple Silicon specific optimizations
    if IS_APPLE_SILICON:
        settings["batch_size"] = 5000  # Apple Silicon handles larger batches well
        settings["cores"] = 8  # M-series chips typically have 8 performance cores
    elif CUDA_AVAILABLE:
        settings["batch_size"] = 10000  # CUDA GPU handles large batches well
    
    return settings

def print_hardware_info():
    """Print detected hardware information"""
    print("\n=== Hardware Detection ===")
    print(f"CPU: {platform.processor() or 'Unknown'}")
    print(f"Cores: {NUM_CORES}")
    print(f"Memory: {AVAILABLE_MEMORY_GB:.1f} GB")
    print(f"Platform: {platform.system()} {platform.release()}")
    print(f"Apple Silicon: {'Yes' if IS_APPLE_SILICON else 'No'}")
    print(f"CUDA Available: {'Yes' if CUDA_AVAILABLE else 'No'}")

def main():
    parser = argparse.ArgumentParser(description="Auto-optimized simulator for Esqueleto Explosivo 3")
    parser.add_argument("spins", nargs="?", type=int, default=None, 
                        help="Number of spins (default: auto-detect based on hardware)")
    parser.add_argument("--plots", action="store_true", 
                        help="Generate visualization plots (requires matplotlib)")
    args = parser.parse_args()
    
    # Get optimal settings based on hardware
    settings = get_optimal_settings()
    
    # Override with user-provided spins if any
    if args.spins is not None:
        settings["num_spins"] = args.spins
    
    # Create a unique run ID with timestamp
    run_id = f"auto_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Print hardware info
    print_hardware_info()
    
    # Create command with optimal parameters
    cmd = [
        "python", "-m", "simulator.optimized",
        "-n", str(settings["num_spins"]),
        "--batch-size", str(settings["batch_size"]),
        "--cores", str(settings["cores"]),
        "--stats-only",
        "--id", run_id
    ]
    
    # Add additional flags
    if settings["no_gpu"]:
        cmd.append("--no-gpu")
    
    if args.plots:
        cmd.append("--plots")
        cmd.append("--enhanced-stats")
    
    # Print command details
    print("\n=== Running with Optimal Settings ===")
    print(f"Spins: {settings['num_spins']:,}")
    print(f"Batch Size: {settings['batch_size']}")
    print(f"Cores: {settings['cores']}")
    print(f"GPU Acceleration: {'Disabled' if settings['no_gpu'] else 'Enabled'}")
    print(f"Visualization: {'Enabled' if args.plots else 'Disabled'}")
    print(f"Run ID: {run_id}")
    print("\nStarting simulation...")
    
    # Run the command
    subprocess.run(cmd)
    
    print(f"\nSimulation complete! Results saved to simulation_results/summary_{run_id}.txt")
    if args.plots:
        print(f"Plots saved to simulation_results/plots/{run_id}_*.png")

if __name__ == "__main__":
    main()