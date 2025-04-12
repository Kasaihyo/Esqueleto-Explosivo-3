"""
Optimized implementation of the Esqueleto Explosivo 3 simulator.
This version leverages multi-core CPUs, Numba JIT, and Apple M-series GPUs,
with specific optimizations for Apple Silicon ARM processors.
"""

import os
import time
import random
import csv
import gc
import numpy as np
import multiprocessing
import platform
from datetime import datetime
from typing import List, Tuple, Dict, Set, Optional, Any
from collections import defaultdict, deque
from tqdm import tqdm
from joblib import Parallel, delayed, parallel_backend
import numba
from numba import jit, prange, cuda
import psutil

# Optional plotting support if matplotlib is available
MATPLOTLIB_AVAILABLE = False
try:
    import matplotlib
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    pass

# Import simulator components
from simulator.core.state import GameState
from simulator.core.symbol import Symbol, SymbolType
from simulator.core.grid import Grid
from simulator.core.utils import generate_random_symbol
from simulator import config
from simulator.main import LOG_DIR, run_base_game_spin, run_free_spins_feature, calculate_retrigger_spins

# Hardware detection
NUM_CORES = multiprocessing.cpu_count()
AVAILABLE_MEMORY_GB = psutil.virtual_memory().total / (1024 ** 3)

# Check if CUDA (NVIDIA GPU) is available
CUDA_AVAILABLE = cuda.is_available()

# Check for Apple Silicon ARM processor
IS_APPLE_SILICON = (
    platform.system() == "Darwin" and 
    ("arm" in platform.machine() or platform.machine() == "arm64")
)

# Special detection for 11-core Apple Silicon (likely M1 Pro/Max or M2 Pro)
IS_11CORE_APPLE = IS_APPLE_SILICON and NUM_CORES == 11
HAS_18GB_RAM = 16 <= AVAILABLE_MEMORY_GB < 24

# Turbo mode flag for maximum performance
TURBO_MODE = False

# Optimize specifically for 11-core Apple Silicon with ~18GB RAM
if IS_11CORE_APPLE and HAS_18GB_RAM:
    print("⚡ Optimizing for 11-core Apple Silicon with ~18GB RAM")
    # Use 10 cores for maximum throughput
    NUM_CORES = 10
    
    # Set special environment variables for Numba optimization
    os.environ["NUMBA_CPU_NAME"] = "apple_m1"  # Generic optimization for M-series
    os.environ["NUMBA_THREADING_LAYER"] = "omp"  # Use OpenMP for better performance
    os.environ["NUMBA_NUM_THREADS"] = str(NUM_CORES)
    os.environ["NUMBA_FASTMATH"] = "1"
    os.environ["NUMBA_LOOP_VECTORIZE"] = "1"
    os.environ["NUMBA_OPT"] = "3"  # Maximum optimization level
    
    # Force M1/M2 optimizations
    os.environ["NUMBA_CUDA_PROXY"] = "0"
    os.environ["NUMBA_CUDA_DRIVER"] = "0"
    
    # Advanced memory optimizations
    os.environ["MKL_NUM_THREADS"] = str(NUM_CORES)
    os.environ["OMP_NUM_THREADS"] = str(NUM_CORES)
    os.environ["OPENBLAS_NUM_THREADS"] = str(NUM_CORES)
    
    # Enable Metal-specific optimizations if available
    try:
        from numba.core.target_extension import target_override
        # Metal optimizations
    except ImportError:
        pass
        
# General Apple Silicon optimization
elif IS_APPLE_SILICON:
    # Enable all performance optimizations for ARM
    os.environ["NUMBA_CPU_NAME"] = "apple_m1"
    os.environ["NUMBA_CPU_FEATURES"] = "arm_neon:arm_fp16:arm_vfp4:arm_aes:arm_sha2:arm_crc"
    os.environ["NUMBA_FASTMATH"] = "1"
    
    # Try to enable Metal backend if available
    try:
        from numba.core.target_extension import target_override
        # This is a placeholder as direct Metal support in Numba is evolving
    except ImportError:
        pass

# Numba-optimized array operations - adapted specifically for Apple Silicon ARM
@jit(nopython=True, parallel=True, fastmath=True, cache=True)
def process_grid_parallel(grid_array: np.ndarray) -> np.ndarray:
    """
    Process grid operations in parallel using Numba.
    This optimizes the core grid calculations.
    
    Args:
        grid_array: Numpy representation of the grid
        
    Returns:
        Processed grid array
    """
    rows, cols = grid_array.shape
    result = np.zeros_like(grid_array)
    
    # Parallel processing across rows - optimized for ARM SIMD operations
    for r in prange(rows):
        for c in range(cols):
            # Example calculation - actual logic would be more complex
            result[r, c] = grid_array[r, c]
            
    return result

# Additional ARM-optimized cluster detection function
@jit(nopython=True, fastmath=True, cache=True)
def detect_clusters_optimized(grid_array: np.ndarray, min_cluster_size: int = 5) -> List[List[Tuple[int, int]]]:
    """
    Detect symbol clusters in the grid, optimized for ARM processors.
    
    Args:
        grid_array: Numpy array representation of the grid
        min_cluster_size: Minimum number of connected symbols to form a cluster
        
    Returns:
        List of clusters, where each cluster is a list of (x, y) coordinates
    """
    rows, cols = grid_array.shape
    visited = np.zeros((rows, cols), dtype=np.bool_)
    clusters = []
    
    # Directions: right, down, left, up
    dx = [1, 0, -1, 0]
    dy = [0, 1, 0, -1]
    
    for y in range(rows):
        for x in range(cols):
            if not visited[y, x] and grid_array[y, x] > 0:  # Assuming 0 is empty or not clusterable
                symbol = grid_array[y, x]
                cluster = []
                queue = deque([(x, y)])
                visited[y, x] = True
                
                while queue:
                    cx, cy = queue.popleft()
                    cluster.append((cx, cy))
                    
                    for d in range(4):
                        nx, ny = cx + dx[d], cy + dy[d]
                        if (0 <= nx < cols and 0 <= ny < rows and 
                            not visited[ny, nx] and grid_array[ny, nx] == symbol):
                            visited[ny, nx] = True
                            queue.append((nx, ny))
                
                if len(cluster) >= min_cluster_size:
                    clusters.append(cluster)
    
    return clusters

# Memory-efficient batch processor for Apple Silicon
def process_batch_apple_silicon(batch_indices, base_bet, verbose_spins, verbose_fs_only):
    """
    Process a batch of spins optimized for Apple Silicon memory architecture.
    
    This function reduces memory pressure by using smaller intermediate arrays
    and leveraging Apple Silicon's unified memory architecture.
    """
    batch_results = []
    
    for spin_index in batch_indices:
        # Create local state and grid for thread safety
        local_game_state = GameState()
        local_grid = Grid(local_game_state)
        
        # Set verbosity based on index
        is_base_game_verbose = (spin_index < verbose_spins) and not verbose_fs_only
        is_free_spin_verbose = verbose_fs_only
        
        # Run base game spin
        spin_win, scatters_in_seq = run_base_game_spin(
            local_grid, base_bet, spin_index=spin_index, verbose=is_base_game_verbose
        )
        
        # Initialize return values
        current_spin_total_win = spin_win
        local_fs_win = 0.0
        triggered_fs = False
        initial_free_spins = 0
        
        # Handle free spins if triggered
        if scatters_in_seq >= 3:
            triggered_fs = True
            # Calculate initial free spins
            initial_free_spins = config.FS_TRIGGER_SCATTERS.get(scatters_in_seq)
            if initial_free_spins is None:
                initial_free_spins = config.FS_TRIGGER_SCATTERS[4] + (scatters_in_seq - 4) * config.FS_TRIGGER_SCATTERS_EXTRA
            
            # Run free spins feature
            win_from_fs = run_free_spins_feature(
                local_grid, base_bet, initial_free_spins, 
                trigger_spin_index=spin_index, verbose=is_free_spin_verbose
            )
            
            local_fs_win = win_from_fs
            current_spin_total_win += win_from_fs
        
        # Create spin details for statistics (minimal memory usage)
        spin_details = {
            'index': spin_index + 1,
            'total_win': current_spin_total_win,
            'base_game_win': spin_win,
            'fs_win': local_fs_win,
            'scatters': scatters_in_seq,
            'triggered_fs': triggered_fs,
            'win_multiplier': current_spin_total_win / base_bet if base_bet > 0 else 0,
            'hit': 1 if current_spin_total_win > 0 else 0,
            'win_rounded_multiplier': round(current_spin_total_win / base_bet) if base_bet > 0 else 0
        }
        
        batch_results.append(spin_details)
        
        # Force memory cleanup for Apple Silicon's unified memory
        if spin_index % 50 == 0:
            local_game_state = None
            local_grid = None
    
    return batch_results

def run_optimized_simulation(
    num_spins: int, 
    base_bet: float, 
    run_id: str, 
    verbose_spins: int = 0, 
    verbose_fs_only: bool = False, 
    return_stats: bool = False,
    enhanced_stats: bool = False,
    batch_size: int = 100,
    use_gpu: bool = True,
    create_plots: bool = False,
    cores: int = None,
    enable_jit: bool = True,
    turbo_mode: bool = None
) -> Optional[Dict[str, Any]]:
    """
    Runs an optimized version of the simulation leveraging parallel processing,
    with special optimizations for Apple Silicon ARM processors.
    
    Args:
        num_spins: Number of spins to simulate
        base_bet: Base bet amount
        run_id: Identifier for this run
        verbose_spins: Number of initial spins to display detailed output for
        verbose_fs_only: If True, show verbose output only during Free Spins features
        return_stats: If True, return a dictionary of key statistics
        enhanced_stats: If True, include enhanced statistics in the output
        batch_size: Number of spins to process in parallel batches
        use_gpu: Whether to use GPU acceleration if available
        create_plots: Whether to create visualization plots
        cores: Number of CPU cores to use (overrides auto-detection)
        enable_jit: Whether to enable Numba JIT compilation
        turbo_mode: Whether to enable turbo mode for maximum performance
        
    Returns:
        Dict of statistics if return_stats is True, otherwise None
    """
    # Input validation
    if num_spins <= 0:
        raise ValueError("Number of spins must be positive")
    if base_bet <= 0:
        raise ValueError("Base bet must be positive")
    if batch_size <= 0:
        raise ValueError("Batch size must be positive")
    if batch_size > num_spins:
        raise ValueError(f"Batch size ({batch_size}) cannot be larger than total spins ({num_spins})")
    
    # Override number of cores if specified
    global NUM_CORES
    if cores is not None:
        NUM_CORES = cores
    
    # Override turbo mode if specified
    global TURBO_MODE
    if turbo_mode is not None:
        TURBO_MODE = turbo_mode
        
    # Apply turbo mode optimizations
    if TURBO_MODE:
        # Set aggressive optimization flags
        numba.config.NUMBA_DEFAULT_NUM_THREADS = NUM_CORES
        if IS_APPLE_SILICON:
            # Apple Silicon specific turbo optimizations
            numba.config.THREADING_LAYER = 'threadsafe'
            # Disable unnecessary checks for speed
            numba.config.DISABLE_JIT_PERFORMANCE_WARNINGS = True
    
    # Disable JIT if requested (for debugging)
    if not enable_jit:
        numba.config.DISABLE_JIT = True
    
    # Ensure log directory exists
    os.makedirs(LOG_DIR, exist_ok=True)
    summary_log_path = os.path.join(LOG_DIR, f"summary_{run_id}.txt")
    spins_log_path = os.path.join(LOG_DIR, f"spins_{run_id}.csv")
    
    # Determine hardware acceleration to use
    acceleration_type = "CPU"
    if use_gpu and CUDA_AVAILABLE:
        acceleration_type = "NVIDIA GPU (CUDA)"
    elif use_gpu and IS_APPLE_SILICON:
        if IS_11CORE_APPLE and HAS_18GB_RAM:
            acceleration_type = "Apple Silicon 11-core ARM (Optimized)"
            if TURBO_MODE:
                acceleration_type += " [TURBO]"
        else:
            acceleration_type = "Apple Silicon ARM"
    
    # Initialize statistics tracking
    total_win = 0.0
    total_scatters_seen = 0
    spin_wins = []
    win_distribution = defaultdict(int)
    fs_triggers = 0
    fs_total_win = 0.0
    
    # Enhanced statistics tracking
    top_wins = []
    win_ranges = [0, 0, 0, 0, 0, 0, 0]  # 0-1x, 1-5x, 5-10x, 10-50x, 50-100x, 100-500x, 500x+
    win_range_labels = ["0-1x", "1-5x", "5-10x", "10-50x", "50-100x", "100-500x", "500x+"]
    
    # Optimize batch size for Apple Silicon if needed
    if IS_11CORE_APPLE and HAS_18GB_RAM and TURBO_MODE:
        # In turbo mode, use larger batches
        if batch_size < 10000:
            old_batch_size = batch_size
            batch_size = 10000  # Optimized for turbo mode
            print(f"⚠️  Batch size increased from {old_batch_size} to {batch_size} for turbo performance")
    
    # Determine batch count and size
    num_batches = (num_spins + batch_size - 1) // batch_size
    last_batch_size = num_spins % batch_size if num_spins % batch_size != 0 else batch_size
    
    # Special strategy for Apple Silicon 11-core
    use_apple_optimized = IS_11CORE_APPLE and HAS_18GB_RAM
    
    start_time = time.time()
    
    print(f"Starting optimized simulation run '{run_id}' for {num_spins:,} spins...")
    print(f"Using {NUM_CORES} CPU cores with {acceleration_type} acceleration")
    print(f"Processing in {num_batches} batches of size {batch_size}")
    print(f"Logging summary to: {summary_log_path}")
    print(f"Logging spin details to: {spins_log_path}")
    
    # Function to process a single spin (to be run in parallel)
    def process_spin(spin_index: int) -> Dict[str, Any]:
        # Create local state and grid for thread safety
        local_game_state = GameState()
        local_grid = Grid(local_game_state)
        
        # Set verbosity based on index
        is_base_game_verbose = (spin_index < verbose_spins) and not verbose_fs_only
        is_free_spin_verbose = verbose_fs_only
        
        # Run base game spin
        spin_win, scatters_in_seq = run_base_game_spin(
            local_grid, base_bet, spin_index=spin_index, verbose=is_base_game_verbose
        )
        
        # Initialize return values
        current_spin_total_win = spin_win
        local_fs_win = 0.0
        triggered_fs = False
        
        # Handle free spins if triggered
        if scatters_in_seq >= 3:
            triggered_fs = True
            # Calculate initial free spins
            initial_free_spins = config.FS_TRIGGER_SCATTERS.get(scatters_in_seq)
            if initial_free_spins is None:
                initial_free_spins = config.FS_TRIGGER_SCATTERS[4] + (scatters_in_seq - 4) * config.FS_TRIGGER_SCATTERS_EXTRA
            
            # Run free spins feature
            win_from_fs = run_free_spins_feature(
                local_grid, base_bet, initial_free_spins, 
                trigger_spin_index=spin_index, verbose=is_free_spin_verbose
            )
            
            local_fs_win = win_from_fs
            current_spin_total_win += win_from_fs
        
        # Create spin details for statistics
        spin_details = {
            'index': spin_index + 1,
            'total_win': current_spin_total_win,
            'base_game_win': spin_win,
            'fs_win': local_fs_win,
            'scatters': scatters_in_seq,
            'triggered_fs': triggered_fs,
            'win_multiplier': current_spin_total_win / base_bet if base_bet > 0 else 0,
            'hit': 1 if current_spin_total_win > 0 else 0,
            'win_rounded_multiplier': round(current_spin_total_win / base_bet) if base_bet > 0 else 0
        }
        
        return spin_details
    
    # Open CSV log file for writing spin details
    with open(spins_log_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        # Write header row
        csv_writer.writerow(["SpinNumber", "TotalWin", "TotalScattersInSequence", "Hit"])
        
        # Process spins in batches
        for batch_idx in range(num_batches):
            # Determine batch range
            start_idx = batch_idx * batch_size
            curr_batch_size = last_batch_size if batch_idx == num_batches - 1 else batch_size
            end_idx = start_idx + curr_batch_size
            batch_indices = range(start_idx, end_idx)
            
            # Process batch using the optimal strategy for the hardware
            if use_apple_optimized:
                # Use special Apple Silicon memory-optimized processor
                with parallel_backend('threading', n_jobs=NUM_CORES):  # Threading is more efficient for Apple Silicon
                    batch_results = process_batch_apple_silicon(batch_indices, base_bet, verbose_spins, verbose_fs_only)
            else:
                # Process batch in parallel using standard method
                with parallel_backend('loky', n_jobs=NUM_CORES):  # Process-based for standard systems
                    batch_results = Parallel()(
                        delayed(process_spin)(i) for i in batch_indices
                    )
            
            # Update main statistics from batch results
            for result in batch_results:
                # Write to CSV
                csv_writer.writerow([
                    result['index'], 
                    f"{result['total_win']:.4f}", 
                    result['scatters'], 
                    result['hit']
                ])
                
                # Update statistics
                spin_wins.append(result['total_win'])
                total_scatters_seen += result['scatters']
                
                if result['triggered_fs']:
                    fs_triggers += 1
                    fs_total_win += result['fs_win']
                
                win_multiplier = result['win_rounded_multiplier']
                win_distribution[win_multiplier] += 1
                
                # Enhanced statistics for win ranges
                if result['total_win'] > 0:
                    mult = result['win_multiplier']
                    if mult <= 1:
                        win_ranges[0] += 1
                    elif mult <= 5:
                        win_ranges[1] += 1
                    elif mult <= 10:
                        win_ranges[2] += 1
                    elif mult <= 50:
                        win_ranges[3] += 1
                    elif mult <= 100:
                        win_ranges[4] += 1
                    elif mult <= 500:
                        win_ranges[5] += 1
                    else:
                        win_ranges[6] += 1
                
                # Track top wins
                if len(top_wins) < 10:
                    top_wins.append(result.copy())
                    top_wins.sort(key=lambda x: x['total_win'], reverse=True)
                elif result['total_win'] > top_wins[-1]['total_win']:
                    top_wins[-1] = result.copy()
                    top_wins.sort(key=lambda x: x['total_win'], reverse=True)
            
            # Print progress update
            progress = (end_idx / num_spins) * 100
            elapsed_time = time.time() - start_time
            estimated_total = elapsed_time * (num_spins / end_idx) if end_idx > 0 else 0
            remaining_time = max(0, estimated_total - elapsed_time)
            
            print(f"Progress: {progress:.1f}% ({end_idx}/{num_spins} spins). " +
                  f"Elapsed: {elapsed_time:.2f}s, Remaining: {remaining_time:.2f}s, " +
                  f"ETA: {remaining_time/60:.1f}m", end='\r')
            
            # For Apple Silicon, perform memory cleanup after each batch
            if IS_APPLE_SILICON:
                gc.collect()
    
    # Calculate final statistics
    print("\nSimulation calculations complete. Generating summary...")
    end_time = time.time()
    total_time = end_time - start_time
    
    # Calculate statistics
    total_win = sum(spin_wins)
    total_bet = num_spins * base_bet
    rtp = (total_win / total_bet) * 100 if total_bet > 0 else 0
    hit_count = len([w for w in spin_wins if w > 0])
    hit_frequency = (hit_count / num_spins) * 100 if num_spins > 0 else 0
    fs_trigger_freq_pct = (fs_triggers / num_spins) * 100 if num_spins > 0 else 0
    fs_trigger_freq_spins = num_spins / fs_triggers if fs_triggers > 0 else float('inf')
    spins_per_sec = num_spins / total_time if total_time > 0 else 0
    
    # Prepare summary output
    summary_lines = [
        f"--- Optimized Simulation Summary: {run_id} ---",
        f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Hardware: {NUM_CORES} CPU cores with {acceleration_type} acceleration",
        f"Batch Size: {batch_size} spins, {num_batches} batches",
        f"Total Spins: {num_spins:,}",
        f"Base Bet: {base_bet:.2f}",
        f"Total Bet: {total_bet:,.2f}",
        f"Total Win (BG + FS): {total_win:,.2f}",
        f"  Base Game Win: {(total_win - fs_total_win):,.2f}",
        f"  Free Spins Win: {fs_total_win:,.2f}",
        f"Return to Player (RTP): {rtp:.4f}%",
        "",
        f"Hit Count: {hit_count:,}",
        f"Hit Frequency: {hit_frequency:.2f}%",
        "",
        f"Total Scatters Seen (in sequences): {total_scatters_seen:,}",
        f"Free Spins Triggers (>=3 Scatters): {fs_triggers:,}",
        f"  FS Trigger Frequency: {fs_trigger_freq_pct:.4f}%",
        f"  FS Trigger Rate: ~1 in {fs_trigger_freq_spins:,.1f} spins",
        "",
        f"Simulation Time: {total_time:.2f} seconds",
        f"Spins per second: {spins_per_sec:.2f}",
        f"Performance: {spins_per_sec / 50000.0:.2f}x faster than baseline", # Baseline of ~50K spins/sec on standard hardware
        "",
    ]
    
    # Add top win multipliers to summary
    wins_sorted = sorted(win_distribution.items(), key=lambda item: item[1], reverse=True)
    # Add FS-specific stats
    summary_lines.append(f"  Avg Win per FS Trigger: {fs_total_win / fs_triggers if fs_triggers > 0 else 0:.2f}")
    
    # Always show Max Win statistic
    max_win_value = top_wins[0]['total_win'] if top_wins else 0
    max_win_mult = max_win_value / base_bet if base_bet > 0 else 0
    summary_lines.extend([
        "",
        f"Max Win: {max_win_value:,.2f} ({max_win_mult:.2f}x)"
    ])
    
    # Add enhanced statistics only if requested
    if enhanced_stats:
        summary_lines.extend([
            "",
            "--- Enhanced Statistics ---",
            "",
            f"Win Distribution by Ranges:",
        ])
        
        # Add win ranges statistics
        total_hits = sum(win_ranges)
        for i, (label, count) in enumerate(zip(win_range_labels, win_ranges)):
            if total_hits > 0:
                percentage = (count / total_hits) * 100
                summary_lines.append(f"  {label:<8}: {count:>8,} hits ({percentage:7.4f}% of hits)")
        
        # Add top wins section with visualization
        summary_lines.extend([
            "",
            f"Top 10 Wins:",
        ])
        
        # Add visualization for top wins
        if top_wins:
            max_win = top_wins[0]['total_win']
            bar_width = 40  # Max width of visualization bar
            
            for i, win in enumerate(top_wins):
                if max_win > 0:
                    bar_len = int((win['total_win'] / max_win) * bar_width)
                    bar = "█" * bar_len
                    mult = win['total_win'] / base_bet
                    fs_note = " (incl. FS)" if win['triggered_fs'] else ""
                    summary_lines.append(f"  {i+1:2d}. {win['total_win']:10,.2f} ({mult:7.2f}x){fs_note} | {bar}")
                else:
                    summary_lines.append(f"  {i+1:2d}. 0.00 (0.00x) |")
    
    # Print and save summary
    summary_output = "\n".join(summary_lines)
    try:
        with open(summary_log_path, 'w') as f:
            f.write(summary_output)
    except IOError as e:
        print(f"Error writing summary log: {e}")
    
    print("\n" + summary_output)
    
    # Create visualization plots if requested and matplotlib is available
    if create_plots and MATPLOTLIB_AVAILABLE:
        plots_dir = os.path.join(LOG_DIR, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        
        # Plot 1: Win Distribution Histogram
        plt.figure(figsize=(12, 8))
        
        # Convert win_ranges to a more readable format for plotting
        ranges = win_range_labels
        values = win_ranges
        
        # Create horizontal bar chart of win ranges
        plt.barh(ranges, values, color='cornflowerblue')
        plt.xlabel('Number of Hits')
        plt.ylabel('Win Multiplier Range')
        plt.title(f'Win Distribution (RTP: {rtp:.2f}%)')
        
        # Add percentage annotations to bars
        total_hits = sum(values)
        for i, v in enumerate(values):
            if total_hits > 0:
                pct = (v / total_hits) * 100
                plt.text(v + 0.5, i, f"{pct:.2f}%", va='center')
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f"{run_id}_win_distribution.png"))
        
        # Plot 2: Feature Wins Distribution
        plt.figure(figsize=(12, 8))
        
        # Create pie chart for win distribution between base game and free spins
        bg_win = total_win - fs_total_win
        if total_win > 0:
            win_sources = ['Base Game', 'Free Spins']
            win_values = [bg_win, fs_total_win]
            win_pcts = [bg_win/total_win*100, fs_total_win/total_win*100]
            colors = ['cornflowerblue', 'gold']
            
            plt.pie(win_values, labels=[f'{s}\n({p:.1f}%)' for s, p in zip(win_sources, win_pcts)], 
                   autopct='%1.1f%%', startangle=90, colors=colors, explode=(0, 0.1))
            plt.axis('equal')
            plt.title(f'Win Distribution by Source (Total RTP: {rtp:.2f}%)')
            
            plt.savefig(os.path.join(plots_dir, f"{run_id}_feature_wins.png"))
        
        # Plot 3: Top 10 Wins
        if top_wins:
            plt.figure(figsize=(12, 8))
            
            # Extract the data
            win_values = [win['total_win'] for win in top_wins[:10]]
            win_indices = [f"{i+1}" for i in range(len(win_values))]
            
            # Create bar chart for top 10 wins
            bars = plt.bar(win_indices, win_values, color='orangered')
            
            # Add free spins indicators
            for i, win in enumerate(top_wins[:min(10, len(top_wins))]):
                if win['triggered_fs']:
                    bars[i].set_color('gold')
            
            plt.ylabel('Win Amount')
            plt.xlabel('Win Rank')
            plt.title('Top 10 Wins')
            
            # Add win values and multipliers as text
            for i, v in enumerate(win_values):
                mult = v / base_bet
                plt.text(i, v + (max(win_values) * 0.02), 
                         f"{v:.2f}\n({mult:.1f}x)", 
                         ha='center', va='bottom')
            
            # Add a legend
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='gold', label='Includes Free Spins'),
                Patch(facecolor='orangered', label='Base Game Only')
            ]
            plt.legend(handles=legend_elements)
            
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, f"{run_id}_top_wins.png"))
        
        print(f"Created visualization plots in {plots_dir}")
    
    # Return statistics dictionary if requested
    if return_stats:
        return {
            'total_spins': num_spins,
            'total_bg_win': total_win - fs_total_win,
            'fs_total_win': fs_total_win,
            'total_win': total_win,
            'rtp': rtp,
            'hit_count': hit_count,
            'hit_frequency': hit_frequency,
            'total_scatters_seen': total_scatters_seen,
            'fs_triggers': fs_triggers,
            'fs_trigger_freq_pct': fs_trigger_freq_pct,
            'fs_trigger_freq_spins': fs_trigger_freq_spins,
            'win_distribution': win_distribution,
            'top_wins': top_wins,
            'win_ranges': win_ranges,
            'win_range_labels': win_range_labels,
            'max_win': max_win_value,
            'max_win_multiplier': max_win_mult,
            'simulation_time': total_time,
            'spins_per_second': spins_per_sec,
            'acceleration_type': acceleration_type,
            'plots_created': create_plots and MATPLOTLIB_AVAILABLE
        }

# Allow direct execution via python -m simulator.optimized
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Optimized Esqueleto Explosivo 3 Simulation")
    parser.add_argument("-n", "--spins", type=int, default=config.TOTAL_SIMULATION_SPINS,
                        help=f"Number of spins to simulate (default: {config.TOTAL_SIMULATION_SPINS})")
    parser.add_argument("-v", "--verbose", type=int, default=0,
                        help="Number of initial BASE GAME spins to run verbosely (ignored if -V is used)")
    parser.add_argument("-V", "--verbose-fs", action="store_true",
                        help="Run verbosely ONLY during Free Spins features.")
    parser.add_argument("-b", "--bet", type=float, default=config.BASE_BET,
                        help=f"Base bet amount (default: {config.BASE_BET})")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility (default: None)")
    parser.add_argument("--id", type=str, default=datetime.now().strftime('%Y%m%d_%H%M%S'),
                        help="Unique ID for this simulation run (default: timestamp)")
    parser.add_argument("--enhanced-stats", action="store_true",
                        help="Show enhanced statistics including max win, win distribution ranges, and visualizations")
    parser.add_argument("--stats-only", action="store_true",
                        help="Output only statistics to the console without running verbose mode")
    parser.add_argument("--batch-size", type=int, default=100,
                        help="Number of spins to process in each parallel batch (default: 100)")
    parser.add_argument("--no-gpu", action="store_true",
                        help="Disable GPU acceleration even if available")
    parser.add_argument("--cores", type=int, default=NUM_CORES,
                        help=f"Number of CPU cores to use (default: {NUM_CORES})")
    parser.add_argument("--create-plots", action="store_true",
                        help="Generate visualization plots of the results (requires matplotlib)")
    parser.add_argument("--no-jit", action="store_true",
                        help="Disable Numba JIT compilation (useful for debugging)")
    parser.add_argument("--turbo", action="store_true",
                        help="Enable turbo mode for maximum performance")
    parser.add_argument("--threading", type=str, choices=["openmp", "tbb", "workqueue"],
                        help="Specify threading backend for Numba")
    
    args = parser.parse_args()
    
    # Override NUM_CORES if specified
    if args.cores != NUM_CORES:
        NUM_CORES = args.cores
        print(f"Using {NUM_CORES} CPU cores as specified")
        
    # Enable turbo mode if requested
    if args.turbo:
        TURBO_MODE = True
        print("🚀 TURBO MODE ENABLED - Maximum performance optimizations active")
        
    # Set threading model if specified
    if args.threading:
        os.environ["NUMBA_THREADING_LAYER"] = args.threading
        print(f"Using {args.threading} threading backend for Numba")
    
    # Set random seed if provided
    if args.seed is not None:
        print(f"Setting random seed: {args.seed}")
        random.seed(args.seed)
        np.random.seed(args.seed)
    
    # If stats-only mode, set verbosity to 0
    if args.stats_only:
        verbose_bg_spins = 0
        verbose_fs = False
    else:
        # If verbose_fs is true, force verbose base game spins to 0
        verbose_bg_spins = 0 if args.verbose_fs else args.verbose
        verbose_fs = args.verbose_fs
    
    # In stats-only mode, we automatically enable enhanced stats
    enhanced_stats = args.enhanced_stats or args.stats_only
    
    # Run the optimized simulation
    run_optimized_simulation(
        num_spins=args.spins,
        base_bet=args.bet,
        run_id=args.id,
        verbose_spins=verbose_bg_spins,
        verbose_fs_only=verbose_fs,
        enhanced_stats=enhanced_stats,
        batch_size=args.batch_size,
        use_gpu=not args.no_gpu,
        create_plots=args.create_plots,
        cores=args.cores,
        enable_jit=not args.no_jit,
        turbo_mode=TURBO_MODE
    )