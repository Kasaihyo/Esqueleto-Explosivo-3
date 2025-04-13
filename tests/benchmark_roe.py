"""
Benchmark script to compare performance of different ROE calculation methods.
"""

import time
import sys
import random
import numpy as np
from simulator.main import calculate_roe
from simulator.optimized import calculate_roe_optimized
from simulator.core.state import GameState
from simulator.core.grid import Grid
from simulator.main import run_base_game_spin, run_free_spins_feature

def generate_fake_simulation_data(num_spins, target_rtp=95.0):
    """Generate fake simulation data with a target RTP"""
    data = []
    
    # Use simpler distribution to control RTP more precisely
    # Model slots wins: many small wins, few big wins
    win_options = [0.0, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]
    
    # Tune these probabilities to achieve target RTP
    # These probabilities sum to 1 and give ~95% RTP
    win_probs = [0.2, 0.3, 0.25, 0.15, 0.06, 0.03, 0.007, 0.002, 0.001]
    
    for _ in range(num_spins):
        # Generate base game win
        win = np.random.choice(win_options, p=win_probs)
        
        # Determine if this was a free spins win (5% chance)
        is_fs = random.random() < 0.05
        fs_win = 0.0
        
        if is_fs:
            # Free spins wins tend to be higher
            fs_win = np.random.choice(win_options, p=[0.1, 0.15, 0.2, 0.25, 0.15, 0.1, 0.03, 0.01, 0.01])
            total_win = win + fs_win
        else:
            total_win = win
            
        data.append({
            'total_win': total_win,
            'base_game_win': win,
            'fs_win': fs_win,
            'triggered_fs': is_fs,
            'scatters': 3 if is_fs else random.randint(0, 2)
        })
    
    # Verify actual RTP and scale if needed
    actual_rtp = (sum(d['total_win'] for d in data) / num_spins) * 100
    if abs(actual_rtp - target_rtp) > 2.0:
        # If RTP is off by more than 2%, scale all wins to match target
        scale_factor = target_rtp / actual_rtp
        for d in data:
            d['total_win'] *= scale_factor
            d['base_game_win'] *= scale_factor
            d['fs_win'] *= scale_factor
    
    return data

def benchmark_roe_methods(num_spins=10000, num_roe_sims=1000, target_rtp=95.0):
    """
    Benchmark different ROE calculation methods with a given number of spins.
    
    Args:
        num_spins: Number of spins to simulate
        num_roe_sims: Number of ROE simulations to run
        target_rtp: Target RTP for the simulated data (%)
    """
    print(f"Generating simulated data for {num_spins:,} spins with target RTP {target_rtp:.1f}%...")
    sim_data = generate_fake_simulation_data(num_spins, target_rtp=target_rtp)
    
    # Calculate actual simulated RTP
    total_win = sum(d['total_win'] for d in sim_data)
    actual_rtp = (total_win / num_spins) * 100
    print(f"Simulated data RTP: {actual_rtp:.2f}%")
    
    # Benchmark 1: Original ROE calculation
    print("\nBenchmarking original ROE calculation...")
    start_time = time.time()
    median_roe, average_roe = calculate_roe(
        rtp=actual_rtp,
        base_bet_for_sim=1.0,
        num_roe_sims=num_roe_sims
    )
    original_time = time.time() - start_time
    print(f"Original ROE calculation: {original_time:.2f} seconds")
    print(f"  Median ROE: {median_roe}")
    print(f"  Average ROE: {average_roe}")
    
    # Benchmark 2: Optimized ROE using separate simulations
    print("\nBenchmarking optimized ROE with separate simulations...")
    start_time = time.time()
    median_roe, average_roe = calculate_roe_optimized(
        main_simulation_data=[],
        rtp=actual_rtp,
        base_bet_for_sim=1.0,
        use_main_data=False,
        num_roe_sims=num_roe_sims
    )
    optimized_separate_time = time.time() - start_time
    print(f"Optimized ROE (separate sims): {optimized_separate_time:.2f} seconds")
    print(f"  Median ROE: {median_roe}")
    print(f"  Average ROE: {average_roe}")
    
    # Benchmark 3: Optimized ROE using main simulation data
    print("\nBenchmarking optimized ROE using main simulation data...")
    start_time = time.time()
    median_roe, average_roe = calculate_roe_optimized(
        main_simulation_data=sim_data,
        rtp=actual_rtp,
        base_bet_for_sim=1.0,
        use_main_data=True,
        num_roe_sims=num_roe_sims
    )
    optimized_main_data_time = time.time() - start_time
    print(f"Optimized ROE (main data): {optimized_main_data_time:.2f} seconds")
    print(f"  Median ROE: {median_roe}")
    print(f"  Average ROE: {average_roe}")
    
    # Summary
    print("\nBenchmark Summary:")
    print(f"Original ROE:                {original_time:.2f}s")
    print(f"Optimized ROE (separate):    {optimized_separate_time:.2f}s  " + 
          f"({original_time / optimized_separate_time:.1f}x faster)")
    print(f"Optimized ROE (main data):   {optimized_main_data_time:.2f}s  " + 
          f"({original_time / optimized_main_data_time:.1f}x faster)")
    
    # Output recommendation
    print("\nRECOMMENDATION:")
    if optimized_main_data_time < optimized_separate_time:
        print("→ Use --roe-use-main-data for fastest performance")
    else:
        print("→ Use --roe-separate-sims for fastest performance")

if __name__ == "__main__":
    spins = 10000
    roe_sims = 500
    target_rtp = 95.0  # Default target RTP
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        spins = int(sys.argv[1])
    if len(sys.argv) > 2:
        roe_sims = int(sys.argv[2])
    if len(sys.argv) > 3:
        target_rtp = float(sys.argv[3])
    
    benchmark_roe_methods(spins, roe_sims, target_rtp)