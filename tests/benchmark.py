#!/usr/bin/env python3
"""
Benchmark script for Esqueleto Explosivo 3 simulator.

This script measures and compares the performance of different simulation implementations
across various hardware configurations and parameter settings.
"""

import os
import sys
import time
import argparse
import json
import platform
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Add the parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import simulator modules
from simulator.main import run_simulation
from simulator.optimized import run_optimized_simulation
import run

def run_benchmark(num_spins=10000, batch_sizes=None, cores=None):
    """
    Run benchmarks comparing original and optimized implementations with various configurations.
    
    Args:
        num_spins: Number of spins to simulate
        batch_sizes: List of batch sizes to test
        cores: List of core counts to test
    
    Returns:
        Dictionary with benchmark results
    """
    if batch_sizes is None:
        batch_sizes = [100, 500, 1000, 5000]
    
    if cores is None:
        hw_info = run.get_hardware_info()
        max_cores = hw_info['cpu_count']
        cores = [1, max(2, max_cores//4), max(4, max_cores//2), max_cores]
        cores = sorted(list(set(cores)))  # Remove duplicates and sort
    
    base_bet = 2.0
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"benchmark_{timestamp}"
    
    # Results dictionary
    results = {
        'system_info': {
            'platform': platform.platform(),
            'processor': platform.processor(),
            'python_version': platform.python_version(),
            'hardware_info': run.get_hardware_info()
        },
        'benchmark_params': {
            'num_spins': num_spins,
            'batch_sizes': batch_sizes,
            'cores': cores,
            'base_bet': base_bet
        },
        'results': {
            'original': {},
            'optimized': {},
            'speedup': {}
        }
    }
    
    # Run original implementation as baseline
    print(f"Running original implementation with {num_spins} spins...")
    start_time = time.time()
    orig_result = run_simulation(
        num_spins=num_spins,
        base_bet=base_bet,
        run_id=f"{run_id}_orig",
        return_stats=True
    )
    orig_time = time.time() - start_time
    
    results['results']['original'] = {
        'time': orig_time,
        'stats': orig_result
    }
    
    print(f"Original implementation: {orig_time:.2f} seconds")
    
    # Run optimized implementations with different batch sizes and core counts
    results['results']['optimized'] = {}
    
    for batch_size in batch_sizes:
        results['results']['optimized'][f'batch_{batch_size}'] = {}
        
        for n_cores in cores:
            print(f"Running optimized implementation with batch size {batch_size}, {n_cores} cores...")
            start_time = time.time()
            opt_result = run_optimized_simulation(
                num_spins=num_spins,
                base_bet=base_bet,
                run_id=f"{run_id}_opt_b{batch_size}_c{n_cores}",
                return_stats=True,
                batch_size=batch_size,
                cores=n_cores
            )
            opt_time = time.time() - start_time
            
            results['results']['optimized'][f'batch_{batch_size}'][f'cores_{n_cores}'] = {
                'time': opt_time,
                'stats': opt_result,
                'speedup': orig_time / opt_time
            }
            
            print(f"  Batch size {batch_size}, {n_cores} cores: {opt_time:.2f} seconds (speedup: {orig_time/opt_time:.2f}x)")
    
    # Calculate and store speedups
    results['results']['speedup'] = {}
    
    for batch_key, batch_results in results['results']['optimized'].items():
        batch_size = int(batch_key.split('_')[1])
        results['results']['speedup'][batch_key] = {}
        
        for core_key, core_results in batch_results.items():
            n_cores = int(core_key.split('_')[1])
            speedup = core_results['speedup']
            results['results']['speedup'][batch_key][core_key] = speedup
    
    # Save results to JSON
    os.makedirs('simulation_results', exist_ok=True)
    result_file = f"simulation_results/benchmark_{timestamp}.json"
    with open(result_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    return results

def create_benchmark_plots(results):
    """
    Create benchmark plots from results.
    
    Args:
        results: Dictionary with benchmark results
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs('simulation_results', exist_ok=True)
    
    # Extract data for plots
    batch_sizes = results['benchmark_params']['batch_sizes']
    cores = results['benchmark_params']['cores']
    
    # Plot 1: Speedup vs Cores for each batch size
    plt.figure(figsize=(10, 6))
    
    for batch_size in batch_sizes:
        batch_key = f'batch_{batch_size}'
        if batch_key in results['results']['speedup']:
            speedups = []
            
            for n_cores in cores:
                core_key = f'cores_{n_cores}'
                if core_key in results['results']['speedup'][batch_key]:
                    speedups.append(results['results']['speedup'][batch_key][core_key])
                else:
                    speedups.append(0)
            
            plt.plot(cores, speedups, 'o-', label=f'Batch Size: {batch_size}')
    
    plt.xlabel('Number of Cores')
    plt.ylabel('Speedup (x times faster)')
    plt.title('Speedup vs Number of Cores for Different Batch Sizes')
    plt.grid(True)
    plt.legend()
    plt.savefig(f'simulation_results/benchmark_{timestamp}_speedup_cores.png')
    
    # Plot 2: Speedup vs Batch Size for each core count
    plt.figure(figsize=(10, 6))
    
    for n_cores in cores:
        core_key = f'cores_{n_cores}'
        speedups = []
        
        for batch_size in batch_sizes:
            batch_key = f'batch_{batch_size}'
            if batch_key in results['results']['speedup'] and core_key in results['results']['speedup'][batch_key]:
                speedups.append(results['results']['speedup'][batch_key][core_key])
            else:
                speedups.append(0)
        
        plt.plot(batch_sizes, speedups, 'o-', label=f'Cores: {n_cores}')
    
    plt.xlabel('Batch Size')
    plt.ylabel('Speedup (x times faster)')
    plt.title('Speedup vs Batch Size for Different Core Counts')
    plt.grid(True)
    plt.legend()
    plt.savefig(f'simulation_results/benchmark_{timestamp}_speedup_batch.png')
    
    # Plot 3: Heatmap of speedup
    plt.figure(figsize=(10, 6))
    speedup_matrix = np.zeros((len(batch_sizes), len(cores)))
    
    for i, batch_size in enumerate(batch_sizes):
        batch_key = f'batch_{batch_size}'
        for j, n_cores in enumerate(cores):
            core_key = f'cores_{n_cores}'
            if batch_key in results['results']['speedup'] and core_key in results['results']['speedup'][batch_key]:
                speedup_matrix[i, j] = results['results']['speedup'][batch_key][core_key]
    
    plt.imshow(speedup_matrix, interpolation='nearest', cmap='hot')
    plt.colorbar(label='Speedup (x times faster)')
    plt.xticks(range(len(cores)), cores)
    plt.yticks(range(len(batch_sizes)), batch_sizes)
    plt.xlabel('Number of Cores')
    plt.ylabel('Batch Size')
    plt.title('Speedup Heatmap: Batch Size vs Number of Cores')
    
    # Add text annotations
    for i in range(len(batch_sizes)):
        for j in range(len(cores)):
            plt.text(j, i, f'{speedup_matrix[i, j]:.1f}x', 
                    ha="center", va="center", color="white" if speedup_matrix[i, j] > 3 else "black")
    
    plt.savefig(f'simulation_results/benchmark_{timestamp}_heatmap.png')
    
    # Plot 4: Bar chart comparing original vs best optimized
    plt.figure(figsize=(10, 6))
    
    # Find best optimized configuration
    best_time = float('inf')
    best_config = None
    
    for batch_key, batch_results in results['results']['optimized'].items():
        for core_key, core_results in batch_results.items():
            if core_results['time'] < best_time:
                best_time = core_results['time']
                best_config = f"{batch_key}, {core_key}"
    
    # Create bar chart
    orig_time = results['results']['original']['time']
    plt.bar(['Original', f'Optimized\n({best_config})'], [orig_time, best_time])
    plt.ylabel('Execution Time (seconds)')
    plt.title('Original vs Best Optimized Implementation')
    
    # Add text annotations
    plt.text(0, orig_time/2, f'{orig_time:.1f}s', ha='center', va='center')
    plt.text(1, best_time/2, f'{best_time:.1f}s\n({orig_time/best_time:.1f}x faster)', ha='center', va='center')
    
    plt.savefig(f'simulation_results/benchmark_{timestamp}_comparison.png')
    
    print(f"Benchmark plots saved to simulation_results/ directory")

def main():
    """Main function to parse arguments and run benchmarks."""
    parser = argparse.ArgumentParser(description='Benchmark the Esqueleto Explosivo 3 simulator')
    parser.add_argument('--spins', type=int, default=10000, help='Number of spins to simulate')
    parser.add_argument('--batch-sizes', type=int, nargs='+', default=[100, 500, 1000, 5000], 
                        help='Batch sizes to test')
    parser.add_argument('--cores', type=int, nargs='+', default=None, 
                        help='Number of cores to test (default: auto-detect)')
    parser.add_argument('--no-plots', action='store_true', help='Disable plot generation')
    
    args = parser.parse_args()
    
    print("Running benchmarks...")
    results = run_benchmark(
        num_spins=args.spins,
        batch_sizes=args.batch_sizes,
        cores=args.cores
    )
    
    if not args.no_plots:
        print("Creating benchmark plots...")
        create_benchmark_plots(results)
    
    print("Benchmarks complete!")
    print(f"Results saved to simulation_results/benchmark_*.json")

if __name__ == '__main__':
    main()