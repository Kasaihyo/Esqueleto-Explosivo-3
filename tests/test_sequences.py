import unittest
import sys
import os
from unittest.mock import patch
# Add project root to sys.path to allow absolute imports
# project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# sys.path.insert(0, project_root)

import random
from typing import List, Tuple, Optional, Set, Dict

# Imports from simulator
from simulator.core.grid import Grid
from simulator.core.symbol import Symbol, SymbolType
from simulator.core.state import GameState
from simulator import config

# Helper function to create a grid from names (copied from other test files)
# def create_grid_from_names(names: List[List[Optional[str]]]) -> Grid:
#     rows = len(names)
#     cols = len(names[0]) if rows > 0 else 0
#     state = GameState()
#     grid = Grid(state, rows=rows, cols=cols)
#     for r in range(rows):
#         for c in range(cols):
#             name = names[r][c]
#             symbol = config.SYMBOLS.get(name) if name else config.SYMBOLS["EMPTY"]
#             grid._set_symbol(r, c, symbol)
#     return grid

# Dummy symbols for testing (Moved to setUpClass)
# DUMMY_SYMBOLS: Dict[str, Symbol] = { ... }

class TestBaseGameSequences(unittest.TestCase):

    # @classmethod
    # def setUpClass(cls):
    #     \"\"\"Set up dummy symbols for the test class.\"\"\"
    #     cls.DUMMY_SYMBOLS: Dict[str, Symbol] = {
    #         \"PNK\": Symbol(\"PNK\", SymbolType.LP),
    #         \"GRN\": Symbol(\"GRN\", SymbolType.LP),
    #         \"BLU\": Symbol(\"BLU\", SymbolType.LP),
    #         \"WILD\": Symbol(\"WILD\", SymbolType.WILD),
    #         \"E_WILD\": Symbol(\"E_WILD\", SymbolType.EXPLOSIVO_WILD),
    #         \"SCATTER\": Symbol(\"SCATTER\", SymbolType.SCATTER),
    #         \"EMPTY\": Symbol(\"EMPTY\", SymbolType.EMPTY),
    #     }

    # Removed decorator
    def test_bg_multiplier_starts_at_1x(self):
        """Test Base Game multiplier starts at 1x on a new spin."""
        game_state = GameState()
        game_state.initialize_spin()
        self.assertEqual(game_state.current_multiplier, 1)
        self.assertFalse(game_state.is_free_spins)

    # Removed decorator
    def test_bg_multiplier_increments_on_win(self):
        """Test Base Game multiplier increments after a winning cluster."""
        game_state = GameState()
        game_state.initialize_spin()
        self.assertEqual(game_state.current_multiplier, 1)
        test_symbol = config.SYMBOLS["PINK_SK"]
        clusters_found = [(test_symbol, [(0,0), (0,1), (0,2), (0,3), (0,4)])]
        game_state.update_after_clusters(clusters_found, did_explode=False)
        self.assertEqual(game_state.current_multiplier, 2)
        game_state.update_after_clusters(clusters_found, did_explode=False)
        self.assertEqual(game_state.current_multiplier, 4)
        game_state.update_after_clusters(clusters_found, did_explode=False)
        self.assertEqual(game_state.current_multiplier, 8)
        game_state.update_after_clusters(clusters_found, did_explode=False)
        self.assertEqual(game_state.current_multiplier, 16)
        game_state.update_after_clusters(clusters_found, did_explode=False)
        self.assertEqual(game_state.current_multiplier, 32)

    # Removed decorator
    def test_bg_multiplier_not_increment_on_ew_explosion_only(self):
        """Test Base Game multiplier does not increment if only an EW exploded (no cluster win)."""
        game_state = GameState()
        game_state.initialize_spin()
        game_state.current_multiplier = 4
        clusters_found = []
        game_state.update_after_clusters(clusters_found, did_explode=True)
        self.assertEqual(game_state.current_multiplier, 4)

    # Removed decorator
    def test_bg_multiplier_caps_at_32x(self):
        """Test Base Game multiplier stops incrementing at 32x."""
        game_state = GameState()
        game_state.initialize_spin()
        test_symbol = config.SYMBOLS["PINK_SK"]
        clusters_found = [(test_symbol, [(0,0), (0,1), (0,2), (0,3), (0,4)])]
        for _ in range(5):
            game_state.update_after_clusters(clusters_found, did_explode=False)
        self.assertEqual(game_state.current_multiplier, 32)
        game_state.update_after_clusters(clusters_found, did_explode=False)
        self.assertEqual(game_state.current_multiplier, 32)
        game_state.update_after_clusters([], did_explode=True)
        self.assertEqual(game_state.current_multiplier, 32)

    # Removed decorator
    def test_bg_multiplier_resets_on_next_spin(self):
        """Test Base Game multiplier resets to 1x on the next spin initialization."""
        game_state = GameState()
        game_state.initialize_spin()
        test_symbol = config.SYMBOLS["PINK_SK"]
        clusters_found = [(test_symbol, [(0,0), (0,1), (0,2), (0,3), (0,4)])]
        game_state.update_after_clusters(clusters_found, did_explode=False)
        game_state.update_after_clusters(clusters_found, did_explode=False)
        self.assertEqual(game_state.current_multiplier, 4)
        game_state.initialize_spin()
        self.assertEqual(game_state.current_multiplier, 1)

    # Removed decorator
    def test_scatter_trigger_none(self):
        """Test 0, 1, or 2 scatters do not trigger FS."""
        game_state_0 = GameState()
        game_state_0.initialize_spin()
        triggered_fs_0, spins_won_0 = game_state_0.finalize_spin_sequence()
        self.assertFalse(triggered_fs_0)
        self.assertEqual(spins_won_0, 0)
        game_state_1 = GameState()
        game_state_1.initialize_spin()
        game_state_1.accumulate_scatter((0, 0))
        triggered_fs_1, spins_won_1 = game_state_1.finalize_spin_sequence()
        self.assertFalse(triggered_fs_1)
        self.assertEqual(spins_won_1, 0)
        game_state_2 = GameState()
        game_state_2.initialize_spin()
        game_state_2.accumulate_scatter((0, 0))
        game_state_2.accumulate_scatter((1, 1))
        triggered_fs_2, spins_won_2 = game_state_2.finalize_spin_sequence()
        self.assertFalse(triggered_fs_2)
        self.assertEqual(spins_won_2, 0)

    # Removed decorator
    def test_scatter_trigger_3(self):
        """Test 3 scatters trigger 10 FS."""
        game_state = GameState()
        game_state.initialize_spin()
        game_state.accumulate_scatter((0, 0))
        game_state.accumulate_scatter((1, 1))
        game_state.accumulate_scatter((2, 2))
        triggered_fs, spins_won = game_state.finalize_spin_sequence()
        self.assertTrue(triggered_fs)
        self.assertEqual(spins_won, 10)

    # Removed decorator
    def test_scatter_trigger_4(self):
        """Test 4 scatters trigger 12 FS."""
        game_state = GameState()
        game_state.initialize_spin()
        game_state.accumulate_scatter((0, 0))
        game_state.accumulate_scatter((1, 1))
        game_state.accumulate_scatter((2, 2))
        game_state.accumulate_scatter((3, 3))
        triggered_fs, spins_won = game_state.finalize_spin_sequence()
        self.assertTrue(triggered_fs)
        self.assertEqual(spins_won, 12)

    # Removed decorator
    def test_scatter_trigger_5(self):
        """Test 5 scatters trigger 14 FS (12 + (5-4)*2)."""
        game_state = GameState()
        game_state.initialize_spin()
        game_state.accumulate_scatter((0, 0))
        game_state.accumulate_scatter((1, 1))
        game_state.accumulate_scatter((2, 2))
        game_state.accumulate_scatter((3, 3))
        game_state.accumulate_scatter((4, 4))
        triggered_fs, spins_won = game_state.finalize_spin_sequence()
        self.assertTrue(triggered_fs)
        self.assertEqual(spins_won, 14)

    # Removed decorator
    def test_scatter_trigger_6_plus(self):
        """Test 6 scatters trigger 16 FS (12 + (6-4)*2)."""
        game_state = GameState()
        game_state.initialize_spin()
        game_state.accumulate_scatter((0, 0))
        game_state.accumulate_scatter((1, 1))
        game_state.accumulate_scatter((2, 2))
        game_state.accumulate_scatter((3, 3))
        game_state.accumulate_scatter((4, 4))
        game_state.accumulate_scatter((0, 1))
        triggered_fs, spins_won = game_state.finalize_spin_sequence()
        self.assertTrue(triggered_fs)
        self.assertEqual(spins_won, 16)

# --- Tests for Scatter Accumulation Across Sequence --- #

class TestScatterAccumulation(unittest.TestCase):
    """Tests for scatter accumulation during a spin sequence."""
    
    def test_scatter_accumulation_across_sequence(self):
        """Test scatters counted correctly across initial drop and multiple avalanches."""
        # Setup a grid with some symbols including scatters
        state = GameState()
        grid = Grid(state)
        
        # Set up initial grid with no scatters
        grid.initialize_spin()
        
        # Manually add scatters to simulate landed scatters
        # Replace some existing symbols with scatters
        for r, c in [(0, 0), (2, 2)]:
            grid._set_symbol(r, c, config.SYMBOLS["SCATTER"])
            
        # Count scatters in initial grid and accumulate them
        for r in range(grid.rows):
            for c in range(grid.cols):
                symbol = grid._get_symbol(r, c)
                if symbol and symbol.type == SymbolType.SCATTER:
                    state.accumulate_scatter((r, c))
        
        # Verify we have accumulated 2 scatters so far
        self.assertEqual(len(state.scatters_collected_this_sequence), 2)
        
        # Simulate an avalanche where new scatters land
        # First we simulate a cluster forming and being cleared
        mock_cluster = [(config.SYMBOLS["PINK_SK"], [(1, 1), (1, 2), (1, 3), (1, 4), (2, 4)])]
        mock_cleared_coords, _, _, _ = grid.process_explosions_and_spawns(mock_cluster)
        
        # Apply the avalanche
        grid.apply_avalanche(state)
        
        # Now add a new scatter that "landed" during the avalanche
        new_scatter_pos = (1, 1)
        grid._set_symbol(new_scatter_pos[0], new_scatter_pos[1], config.SYMBOLS["SCATTER"])
        grid.landed_coords.add(new_scatter_pos)
        
        # Accumulate this newly landed scatter
        symbol = grid._get_symbol(new_scatter_pos[0], new_scatter_pos[1])
        if symbol and symbol.type == SymbolType.SCATTER:
            state.accumulate_scatter(new_scatter_pos)
            
        # Verify we now have a total of 3 scatters
        self.assertEqual(len(state.scatters_collected_this_sequence), 3)
        
        # Simulate another avalanche with more scatters landing
        mock_cluster2 = [(config.SYMBOLS["GREEN_SK"], [(3, 0), (3, 1), (3, 2), (3, 3), (3, 4)])]
        mock_cleared_coords2, _, _, _ = grid.process_explosions_and_spawns(mock_cluster2)
        
        # Apply second avalanche
        grid.apply_avalanche(state)
        
        # Add two more scatters that "landed" during the second avalanche
        new_scatter_pos2 = [(3, 2), (3, 3)]
        for pos in new_scatter_pos2:
            grid._set_symbol(pos[0], pos[1], config.SYMBOLS["SCATTER"])
            grid.landed_coords.add(pos)
            
            # Accumulate this newly landed scatter
            symbol = grid._get_symbol(pos[0], pos[1])
            if symbol and symbol.type == SymbolType.SCATTER:
                state.accumulate_scatter(pos)
                
        # Verify we now have a total of 5 scatters
        self.assertEqual(len(state.scatters_collected_this_sequence), 5)
        
        # Finalize the spin sequence
        triggered_fs, spins_won = state.finalize_spin_sequence()
        
        # Verify free spins were triggered with the correct number of spins
        self.assertTrue(triggered_fs)
        self.assertEqual(spins_won, 14)  # 12 base + (5-4)*2 = 14 spins
        
    def test_scatter_deduplication(self):
        """Test that duplicate scatter coordinates aren't counted multiple times."""
        state = GameState()
        
        # Accumulate the same scatter coordinate multiple times
        scatter_pos = (2, 3)
        state.accumulate_scatter(scatter_pos)
        state.accumulate_scatter(scatter_pos)
        state.accumulate_scatter(scatter_pos)
        
        # Should only count as 1 scatter
        self.assertEqual(len(state.scatters_collected_this_sequence), 1)
        
        # Add a different scatter coordinate
        state.accumulate_scatter((1, 1))
        
        # Should now be 2 scatters
        self.assertEqual(len(state.scatters_collected_this_sequence), 2)

if __name__ == '__main__':
    unittest.main() 