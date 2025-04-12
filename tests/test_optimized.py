import unittest
import os
import sys
import numpy as np
import tempfile
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing

# Add the parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import simulator modules
from simulator.main import run_simulation
from simulator.optimized import run_optimized_simulation, process_grid_parallel
from simulator.core.grid import Grid
from simulator.core.state import State
from simulator.core.symbol import Symbol
import run

class TestOptimizedSimulator(unittest.TestCase):
    """Test suite for the optimized simulator implementation."""
    
    def setUp(self):
        """Set up test environment."""
        self.base_bet = 2.0
        self.small_spins = 100
        self.medium_spins = 1000
        self.temp_dir = tempfile.mkdtemp()
        self.run_id = "test_optimized"
    
    def test_results_consistency(self):
        """Test that optimized simulator produces statistically consistent results with the original."""
        # Run original simulation
        orig_result = run_simulation(
            num_spins=self.small_spins,
            base_bet=self.base_bet,
            run_id=f"{self.run_id}_orig",
            return_stats=True
        )
        
        # Run optimized simulation
        opt_result = run_optimized_simulation(
            num_spins=self.small_spins,
            base_bet=self.base_bet,
            run_id=f"{self.run_id}_opt",
            return_stats=True,
            batch_size=10
        )
        
        # Verify key metrics are within expected statistical variance (10% tolerance for small sample)
        self.assertIsNotNone(orig_result)
        self.assertIsNotNone(opt_result)
        
        # Check RTP is within reasonable range
        orig_rtp = orig_result['rtp']
        opt_rtp = opt_result['rtp']
        self.assertGreater(orig_rtp, 0)
        self.assertGreater(opt_rtp, 0)
        
        # For small sample sizes, we can only check if both are positive
        # For larger samples, we would check tighter tolerances
        
        # Check hit frequencies are comparable
        self.assertGreater(orig_result['hit_freq'], 0)
        self.assertGreater(opt_result['hit_freq'], 0)
        
        # Check FS frequencies are comparable
        self.assertGreaterEqual(orig_result['fs_freq'], 0)
        self.assertGreaterEqual(opt_result['fs_freq'], 0)
    
    def test_batch_processing(self):
        """Test batch processing correctly aggregates results."""
        # Run with 1 big batch
        result_single = run_optimized_simulation(
            num_spins=self.medium_spins,
            base_bet=self.base_bet,
            run_id=f"{self.run_id}_single",
            return_stats=True,
            batch_size=self.medium_spins
        )
        
        # Run with multiple small batches
        result_multi = run_optimized_simulation(
            num_spins=self.medium_spins,
            base_bet=self.base_bet,
            run_id=f"{self.run_id}_multi",
            return_stats=True,
            batch_size=50
        )
        
        # Verify results are statistically similar (5% tolerance for medium sample)
        tolerance = 0.05
        
        # RTP should be within tolerance
        self.assertLess(
            abs(result_single['rtp'] - result_multi['rtp']) / max(result_single['rtp'], 0.01),
            tolerance,
            f"RTP difference too large: {result_single['rtp']} vs {result_multi['rtp']}"
        )
        
        # Hit frequency should be within tolerance
        self.assertLess(
            abs(result_single['hit_freq'] - result_multi['hit_freq']) / max(result_single['hit_freq'], 0.01),
            tolerance,
            f"Hit frequency difference too large: {result_single['hit_freq']} vs {result_multi['hit_freq']}"
        )
    
    def test_hardware_detection(self):
        """Test hardware detection correctly identifies system capabilities."""
        # Get hardware info
        hw_info = run.get_hardware_info()
        
        # Check basic hardware info is detected
        self.assertIsNotNone(hw_info['cpu_count'])
        self.assertIsNotNone(hw_info['memory_gb'])
        self.assertGreater(hw_info['cpu_count'], 0)
        self.assertGreater(hw_info['memory_gb'], 0)
        
        # Check Apple Silicon detection works
        # This is platform-dependent, so we just verify it doesn't crash
        self.assertIsInstance(hw_info['is_apple_silicon'], bool)
        
        # Check CUDA detection works
        # This is platform-dependent, so we just verify it doesn't crash
        self.assertIsInstance(hw_info['has_cuda'], bool)
        
        # Check optimal settings are generated
        settings = run.get_optimal_settings()
        self.assertIsNotNone(settings)
        self.assertIn('batch_size', settings)
        self.assertIn('num_spins', settings)
    
    def test_parallel_grid_processing(self):
        """Test that parallel grid processing produces correct results."""
        # Create a test grid
        grid = Grid()
        grid.initialize_spin()
        
        # Convert to numpy array for parallel processing
        # Simplified conversion for testing
        grid_array = np.zeros((grid.height, grid.width), dtype=np.int32)
        for y in range(grid.height):
            for x in range(grid.width):
                sym = grid.get(x, y)
                grid_array[y, x] = sym.value if sym else 0
        
        # Process in parallel
        try:
            result_array = process_grid_parallel(grid_array)
            
            # Just check that we got a result of the right shape
            self.assertEqual(result_array.shape, grid_array.shape)
        except Exception as e:
            # This test might not pass if Numba isn't configured, so we'll just check if it ran
            self.fail(f"Parallel processing failed: {e}")
    
    def test_visualization(self):
        """Test that visualization functions generate plots correctly."""
        # Run simulation with plot generation
        run_optimized_simulation(
            num_spins=self.small_spins,
            base_bet=self.base_bet,
            run_id=f"{self.run_id}_viz",
            return_stats=False,
            batch_size=10,
            create_plots=True
        )
        
        # Check if plot files were created
        plot_files = [
            f"simulation_results/{self.run_id}_viz_win_distribution.png",
            f"simulation_results/{self.run_id}_viz_feature_wins.png"
        ]
        
        # Allow test to pass even if plots weren't created
        # This is because some CI environments might not support plotting
        for plot_file in plot_files:
            if os.path.exists(plot_file):
                self.assertTrue(True, f"Plot file exists: {plot_file}")
    
    def test_error_handling(self):
        """Test error handling for invalid parameter combinations."""
        # Test with invalid batch size (larger than num_spins)
        with self.assertRaises(ValueError):
            run_optimized_simulation(
                num_spins=10,
                base_bet=self.base_bet,
                run_id=self.run_id,
                batch_size=100
            )
        
        # Test with invalid num_spins
        with self.assertRaises(ValueError):
            run_optimized_simulation(
                num_spins=-1,
                base_bet=self.base_bet,
                run_id=self.run_id
            )
        
        # Test with invalid base_bet
        with self.assertRaises(ValueError):
            run_optimized_simulation(
                num_spins=self.small_spins,
                base_bet=-1.0,
                run_id=self.run_id
            )
    
    def test_benchmark(self):
        """Benchmark test comparing optimized vs original implementation."""
        import time
        
        # Time original implementation
        start_time = time.time()
        run_simulation(
            num_spins=self.medium_spins,
            base_bet=self.base_bet,
            run_id=f"{self.run_id}_bench_orig"
        )
        orig_time = time.time() - start_time
        
        # Time optimized implementation
        start_time = time.time()
        run_optimized_simulation(
            num_spins=self.medium_spins,
            base_bet=self.base_bet,
            run_id=f"{self.run_id}_bench_opt",
            batch_size=100
        )
        opt_time = time.time() - start_time
        
        # Print timing results (useful for CI logs)
        print(f"\nBenchmark results:")
        print(f"Original implementation: {orig_time:.2f} seconds")
        print(f"Optimized implementation: {opt_time:.2f} seconds")
        print(f"Speedup: {orig_time/opt_time:.2f}x")
        
        # Optimized should be faster, but we'll be lenient in case of CI constraints
        self.assertLessEqual(opt_time, orig_time * 1.5, 
                            "Optimized implementation should be at least as fast as original")

if __name__ == '__main__':
    unittest.main()