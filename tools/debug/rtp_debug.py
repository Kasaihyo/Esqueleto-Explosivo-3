#!/usr/bin/env python3
"""
Debug script for investigating RTP differences between main and optimized implementations.
"""

import sys
import random
import numpy as np
from pathlib import Path

# Add the parent directory to sys.path to allow imports from simulator
sys.path.insert(0, str(Path(__file__).parent.parent))

from simulator.main import run_base_game_spin as main_run_spin
from simulator.optimized import run_optimized_simulation, process_batch_apple_silicon
from simulator.core.state import GameState
from simulator.core.grid import Grid

def investigate_rtp_difference(seed=111, num_spins=5):
    """
    Compare the exact sequence of events between main and optimized implementations
    with identical seeds.
    """
    print(f"\n==== RTP Investigation with Seed={seed}, Spins={num_spins} ====\n")
    
    # First, run a few spins with the main implementation
    print("\n[MAIN IMPLEMENTATION TEST]")
    random.seed(seed)
    np.random.seed(seed)
    
    # Initial RNG check
    initial_randoms = [random.random() for _ in range(3)]
    print(f"Main Implementation - Initial random numbers: {initial_randoms}")
    random.seed(seed)  # Reset the seed
    np.random.seed(seed)
    
    main_game_state = GameState()
    main_grid = Grid(main_game_state)
    
    main_total_win = 0.0
    base_bet = 1.0
    
    # Run spins in main implementation with debug logging enabled
    for i in range(num_spins):
        spin_win, scatters = main_run_spin(main_grid, base_bet, spin_index=i, verbose=False, debug_rtp=True)
        main_total_win += spin_win
        print(f"[MAIN SUMMARY] Spin {i+1}: Win={spin_win:.2f}, Total={main_total_win:.2f}")
    
    print(f"\nMain Implementation - Total win after {num_spins} spins: {main_total_win:.2f}")
    main_rtp = (main_total_win / (num_spins * base_bet)) * 100
    print(f"Main Implementation - RTP: {main_rtp:.2f}%")
    
    # Next, run the optimized implementation with identical seed but with forced sequential execution
    print("\n\n[OPTIMIZED IMPLEMENTATION TEST - SEQUENTIAL MODE]")
    
    # Reset the RNG state
    random.seed(seed)
    np.random.seed(seed)
    
    # Check initial RNG state
    initial_randoms = [random.random() for _ in range(3)]
    print(f"Optimized Implementation - Initial random numbers: {initial_randoms}")
    random.seed(seed)  # Reset the seed
    np.random.seed(seed)
    
    # Create a batch of spin indices - just sequential for now
    batch_indices = list(range(num_spins))
    
    # Directly run the batch processor in sequential mode
    print("\nRunning sequential batch processing...")
    opt_game_state = GameState()
    opt_grid = Grid(opt_game_state)
    
    opt_total_win = 0.0
    
    # Set the RNG state once at the beginning
    random.seed(seed)
    np.random.seed(seed)
    initial_state = random.getstate()
    
    # Simulate the sequential execution by first running previous spins
    # to properly advance the RNG state
    print(f"[OPT-DEBUG] Preparing the RNG state as if executing {len(batch_indices)} spins sequentially...")
    
    # Create a temporary grid to run through the previous spins just to advance RNG
    if batch_indices[0] > 0:
        temp_game_state = GameState()
        temp_grid = Grid(temp_game_state)
        for i in range(batch_indices[0]):
            # Use a temporary grid to advance the RNG to the proper state
            # These results are discarded - we just need to sync RNG state
            if i == 0:
                # Save initial RNG state for first spin
                initial_samples = [random.random() for _ in range(3)]
                random.setstate(initial_state)
                np.random.seed(seed)
                print(f"[OPT-DEBUG] Initial RNG samples: {initial_samples}")
            
            # Run the spin (but discard results) to advance RNG
            main_run_spin(temp_grid, base_bet, spin_index=i, verbose=False, debug_rtp=False)
            
            if i % 10 == 0 and i > 0:
                print(f"[OPT-DEBUG] Advanced RNG state through {i} spins...")
    
    # Manually run each spin to closely track the execution flow
    for spin_index in batch_indices:
        # Debug check (don't reset the seed)
        rng_samples = [random.random() for _ in range(3)]
        print(f"[OPT-DEBUG] Spin {spin_index + 1}: Using proper sequential RNG state, samples: {rng_samples}")
        # Store RNG state before this spin
        current_rng_state = random.getstate()
        
        # Run the spin with verbose debugging
        spin_win, scatters = main_run_spin(opt_grid, base_bet, spin_index=spin_index, verbose=False, debug_rtp=True)
        opt_total_win += spin_win
        print(f"[OPT SUMMARY] Spin {spin_index+1}: Win={spin_win:.2f}, Total={opt_total_win:.2f}")
    
    print(f"\nOptimized Implementation (Sequential) - Total win after {num_spins} spins: {opt_total_win:.2f}")
    opt_rtp = (opt_total_win / (num_spins * base_bet)) * 100
    print(f"Optimized Implementation (Sequential) - RTP: {opt_rtp:.2f}%")
    
    # Compare the results
    print("\n==== COMPARISON ====")
    print(f"Main RTP: {main_rtp:.2f}%")
    print(f"Optimized RTP: {opt_rtp:.2f}%")
    print(f"Difference: {abs(main_rtp - opt_rtp):.2f} percentage points")
    if abs(main_rtp - opt_rtp) < 0.01:
        print("MATCH! The RTP values are practically identical.")
    else:
        print("MISMATCH! The RTP values differ significantly.")

if __name__ == "__main__":
    # Run the investigation with a specific seed
    investigate_rtp_difference(seed=111, num_spins=5)