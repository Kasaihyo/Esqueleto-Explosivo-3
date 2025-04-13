#!/usr/bin/env python
"""
Auto-optimized simulator launcher for Esqueleto Explosivo 3.
This script automatically detects hardware capabilities and runs the simulator
with optimal settings, with special optimizations for Apple Silicon ARM processors.

Usage:
    python run.py [--spins SPINS] [--plots] [--profile PROFILE]

Example:
    python run.py                  # Run with auto-detected optimal settings
    python run.py --spins 100000   # Run 100,000 spins
    python run.py --profile m1     # Use optimization profile for M1 chip
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

def get_hardware_info():
    """Get hardware information as a dictionary"""
    return {
        "cpu_count": NUM_CORES,
        "memory_gb": AVAILABLE_MEMORY_GB,
        "is_apple_silicon": IS_APPLE_SILICON,
        "has_cuda": CUDA_AVAILABLE,
        "platform": f"{platform.system()} {platform.release()}",
        "processor": platform.processor() or ("arm" if IS_APPLE_SILICON else "unknown")
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
    settings["no_gpu"] = not (CUDA_AVAILABLE or IS_APPLE_SILICON)  # Use GPU if available
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
        if profile == "m1":
            # M1 chip (8-core, typically 8GB/16GB memory)
            settings["cores"] = min(7, NUM_CORES)
            settings["batch_size"] = 3000
            settings["enable_jit"] = True
        elif profile == "m1pro":
            # M1 Pro (8-10 cores, typically 16GB memory)
            settings["cores"] = min(8, NUM_CORES)
            settings["batch_size"] = 6000
            settings["enable_jit"] = True
        elif profile == "m1max":
            # M1 Max (10 cores, typically 32GB memory)
            settings["cores"] = min(8, NUM_CORES)
            settings["batch_size"] = 8000
            settings["enable_jit"] = True
        elif profile == "m2":
            # M2 (8 cores, typically 8GB/16GB/24GB memory)
            settings["cores"] = min(7, NUM_CORES)
            settings["batch_size"] = 4000
            settings["enable_jit"] = True
        elif profile == "m2pro":
            # M2 Pro (10-12 cores, typically 16GB/32GB memory)
            settings["cores"] = min(8, NUM_CORES)
            settings["batch_size"] = 6000
            settings["enable_jit"] = True
        elif profile == "m3":
            # M3 or newer
            settings["cores"] = min(NUM_CORES - 2, NUM_CORES)
            settings["batch_size"] = 8000
            settings["enable_jit"] = True
        elif profile == "11core18gb":
            # Exact match for the detected hardware (11 cores, 18GB RAM)
            settings["cores"] = 10  # Use 10 cores for maximum throughput
            settings["batch_size"] = 10000  # Larger batch size for better throughput
            settings["num_spins"] = 1_000_000
            settings["enable_jit"] = True
            # Add Apple Silicon specific flags
            settings["apple_silicon_turbo"] = True
            settings["threading_model"] = "openmp"  # Use OpenMP for better threading performance
    
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
    parser = argparse.ArgumentParser(description="Auto-optimized simulator for Esqueleto Explosivo 3")
    parser.add_argument("--spins", type=int, default=None, 
                        help="Number of spins (default: auto-detect based on hardware)")
    parser.add_argument("--plots", action="store_true", 
                        help="Generate visualization plots (requires matplotlib)")
    parser.add_argument("--profile", type=str, choices=["m1", "m1pro", "m1max", "m2", "m2pro", "m3", "11core18gb"], 
                        help="Optimization profile for specific hardware")
    parser.add_argument("--debug", action="store_true",
                        help="Run with enhanced debugging output")
    parser.add_argument("--bet", type=float, default=1.0,
                        help="Base bet amount (default: 1.0)")
    parser.add_argument("--roe", action="store_true", dest="calculate_roe", default=True,
                        help="Calculate ROE statistics (default: enabled)")
    parser.add_argument("--no-roe", action="store_false", dest="calculate_roe",
                        help="Disable ROE calculation")
    parser.add_argument("--roe-use-main-data", action="store_true", default=True,
                        help="Use main simulation data for ROE calculation (faster)")
    parser.add_argument("--roe-separate-sims", action="store_false", dest="roe_use_main_data",
                        help="Run separate simulations for ROE calculation (slower)")
    parser.add_argument("--roe-num-sims", type=int, default=1000,
                        help="Number of ROE simulations to run if not using main data (default: 1000)")
    args = parser.parse_args()
    
    # Get optimal settings based on hardware and profile
    settings = get_optimal_settings(args.profile)
    
    # If 11-core device with ~18GB RAM is detected, suggest the specific profile
    hw_info = get_hardware_info()
    if hw_info["cpu_count"] == 11 and 16 <= hw_info["memory_gb"] < 24 and not args.profile:
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
        "python", "-m", "simulator.optimized",
        "-n", str(settings["num_spins"]),
        "--batch-size", str(settings["batch_size"]),
        "--cores", str(settings["cores"]),
        "--stats-only",
        "--id", run_id,
        "--bet", str(args.bet)
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
    
    # Print command details
    print("\n=== Running with Optimal Settings ===")
    print(f"Spins: {settings['num_spins']:,}")
    print(f"Batch Size: {settings['batch_size']}")
    print(f"Cores: {settings['cores']}")
    print(f"GPU Acceleration: {'Disabled' if settings['no_gpu'] else 'Enabled'}")
    print(f"JIT Compilation: {'Enabled' if settings['enable_jit'] else 'Disabled'}")
    print(f"Visualization: {'Enabled' if args.plots else 'Disabled'}")
    print(f"Turbo Mode: {'Enabled' if settings.get('apple_silicon_turbo', False) else 'Disabled'}")
    print(f"Threading Model: {settings.get('threading_model', 'default')}")
    
    # Print ROE settings
    roe_status = "Disabled" if not args.calculate_roe else "Enabled"
    roe_method = "Using main data (fast)" if args.roe_use_main_data else f"Separate sims ({args.roe_num_sims})"
    print(f"ROE Calculation: {roe_status} - {roe_method if args.calculate_roe else ''}")
    
    print(f"Run ID: {run_id}")
    print("\nStarting simulation...")
    
    # Run the command
    subprocess.run(cmd)
    
    print(f"\nSimulation complete! Results saved to simulation_results/summary_{run_id}.txt")
    if args.plots:
        print(f"Plots saved to simulation_results/plots/{run_id}_*.png")

if __name__ == "__main__":
    main()