import unittest
from unittest.mock import patch, MagicMock, mock_open
import tempfile
import os
import random
import csv
import io
import sys
from contextlib import redirect_stdout

# Import the optimized simulator
from simulator.optimized import run_optimized_simulation
from simulator import config

class TestOptimizedIntegration(unittest.TestCase):
    """Test the integration of the optimized simulator."""
    
    def setUp(self):
        """Set up fixed random seed for reproducibility."""
        random.seed(42)
        # Create temp directory for outputs
        self.temp_dir = tempfile.TemporaryDirectory()
        self.original_log_dir = os.environ.get("SIMULATOR_LOG_DIR")
        os.environ["SIMULATOR_LOG_DIR"] = self.temp_dir.name
        
    def tearDown(self):
        """Clean up test environment."""
        # Restore original log directory
        if self.original_log_dir:
            os.environ["SIMULATOR_LOG_DIR"] = self.original_log_dir
        else:
            os.environ.pop("SIMULATOR_LOG_DIR", None)
            
        # Clean up temp directory
        self.temp_dir.cleanup()
        
    def test_basic_simulation(self):
        """Test that the optimized simulator runs and produces statistics."""
        # Run a small simulation for testing
        results = run_optimized_simulation(
            num_spins=10,
            base_bet=1.0,
            run_id="test_basic",
            batch_size=5,  # Ensure batch_size < num_spins
            return_stats=True,
            calculate_roe=False
        )
        
        # Basic validations
        self.assertEqual(results['total_spins'], 10)
        self.assertGreaterEqual(results['rtp'], 0)
        self.assertTrue(0 <= results['hit_frequency'] <= 100)
        
    def test_log_files_creation(self):
        """Test that log files are created correctly."""
        run_id = "test_logs"
        
        # Run a simulation
        with patch('simulator.optimized.LOG_DIR', self.temp_dir.name):
            run_optimized_simulation(
                num_spins=5,
                base_bet=1.0,
                run_id=run_id,
                batch_size=2,  # Ensure batch_size < num_spins
                calculate_roe=False
            )
            
        # Check that summary and spins files were created
        summary_path = os.path.join(self.temp_dir.name, f"summary_{run_id}.txt")
        spins_path = os.path.join(self.temp_dir.name, f"spins_{run_id}.csv")
        
        self.assertTrue(os.path.exists(summary_path), f"Summary file {summary_path} not created")
        self.assertTrue(os.path.exists(spins_path), f"Spins CSV file {spins_path} not created")
        
        # Check that summary file contains expected information
        with open(summary_path, 'r') as f:
            summary_content = f.read()
            self.assertIn("Optimized Simulation Summary", summary_content)
            self.assertIn("Return to Player (RTP)", summary_content)
            
        # Check that CSV file has expected structure
        with open(spins_path, 'r') as f:
            csv_reader = csv.reader(f)
            header = next(csv_reader)
            
            # Check header contains expected columns
            self.assertIn("SpinNumber", header)
            self.assertIn("TotalWin", header)
            self.assertIn("Hit", header)
            
    def test_roe_calculation(self):
        """Test that ROE calculation works correctly."""
        # Test with ROE enabled, using main data
        results_with_roe = run_optimized_simulation(
            num_spins=10,
            base_bet=1.0,
            run_id="test_roe",
            batch_size=5,  # Ensure batch_size < num_spins
            return_stats=True,
            calculate_roe=True,
            roe_use_main_data=True,
            roe_num_sims=10  # Small value for testing
        )
        
        # ROE should be calculated
        self.assertIn('median_roe', results_with_roe)
        self.assertIn('average_roe', results_with_roe)
        
        # Test with ROE disabled
        results_without_roe = run_optimized_simulation(
            num_spins=10,
            base_bet=1.0,
            run_id="test_no_roe",
            batch_size=5,  # Ensure batch_size < num_spins
            return_stats=True,
            calculate_roe=False
        )
        
        # ROE should be N/A
        self.assertEqual(results_without_roe['median_roe'], "N/A")
        self.assertEqual(results_without_roe['average_roe'], "N/A")
        
    def test_multi_core_processing(self):
        """Test that multi-core processing works correctly."""
        # Test with a single core
        single_core_results = run_optimized_simulation(
            num_spins=10,
            base_bet=1.0,
            run_id="test_single_core",
            batch_size=5,  # Ensure batch_size < num_spins
            cores=1,
            return_stats=True,
            calculate_roe=False
        )
        
        # Test with multiple cores (minimum 2)
        multi_core_results = run_optimized_simulation(
            num_spins=10,
            base_bet=1.0,
            run_id="test_multi_core",
            batch_size=5,  # Ensure batch_size < num_spins
            cores=2,
            return_stats=True,
            calculate_roe=False
        )
        
        # Both should run successfully and return the correct number of spins
        self.assertEqual(single_core_results['total_spins'], 10)
        self.assertEqual(multi_core_results['total_spins'], 10)
        
if __name__ == '__main__':
    unittest.main()