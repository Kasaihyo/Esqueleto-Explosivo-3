import unittest
import sys
import os
from unittest.mock import patch, MagicMock
# Add project root to sys.path to allow absolute imports
# project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# sys.path.insert(0, project_root)

import random
from typing import List, Tuple, Optional, Set, Dict

# Need to import from the simulator module
from simulator.core.grid import Grid
from simulator.core.symbol import Symbol, SymbolType
from simulator.core.state import GameState
from simulator.config import GRID_ROWS, GRID_COLS, SYMBOLS, SYMBOL_GENERATION_WEIGHTS_BG, SYMBOL_GENERATION_WEIGHTS_FS

# Dummy symbols for testing (Moved to setUpClass)
# DUMMY_BG_SYMBOLS: Dict[str, Symbol] = { ... }
# DUMMY_FS_SYMBOLS: Dict[str, Symbol] = { ... }

class TestGridInitialization(unittest.TestCase):
    """Tests for the Grid class initialization and symbol population."""

    # @classmethod
    # def setUpClass(cls):
    #     """Define dummy symbols for testing purposes."""
    #     # These are simplified and don't need actual weights for init tests
    #     cls.DUMMY_SYMBOL_BG_A = Symbol(name="BG_A", type=SymbolType.LP)
    #     cls.DUMMY_SYMBOL_BG_B = Symbol(name="BG_B", type=SymbolType.HP)
    #     cls.DUMMY_SYMBOL_FS_A = Symbol(name="FS_A", type=SymbolType.LP)
    #     cls.DUMMY_SYMBOL_FS_B = Symbol(name="FS_B", type=SymbolType.WILD)

    #     cls.DUMMY_SYMBOLS_DICT_BG = {
    #         "BG_A": cls.DUMMY_SYMBOL_BG_A,
    #         "BG_B": cls.DUMMY_SYMBOL_BG_B,
    #         "EMPTY": SYMBOLS["EMPTY"] # Need EMPTY for grid init
    #     }
    #     cls.DUMMY_WEIGHTS_BG = {
    #         "BG_A": 50,
    #         "BG_B": 50
    #     }

    #     cls.DUMMY_SYMBOLS_DICT_FS = {
    #         "FS_A": cls.DUMMY_SYMBOL_FS_A,
    #         "FS_B": cls.DUMMY_SYMBOL_FS_B,
    #         "EMPTY": SYMBOLS["EMPTY"] # Need EMPTY for grid init
    #     }
    #     cls.DUMMY_WEIGHTS_FS = {
    #         "FS_A": 60,
    #         "FS_B": 40
    #     }

    def setUp(self):
        """Set up a new GameState and Grid for each test."""
        self.state = GameState()
        self.grid = Grid(self.state)

    def test_grid_dimensions(self):
        """Test if the grid is initialized with the correct dimensions."""
        self.assertEqual(len(self.grid.symbols), GRID_ROWS)
        self.assertEqual(len(self.grid.symbols[0]), GRID_COLS)
        # Check all rows have the same number of columns
        for row in self.grid.symbols:
            self.assertEqual(len(row), GRID_COLS)
            # Check initial state is EMPTY
            for symbol in row:
                self.assertEqual(symbol.name, "EMPTY")

    # @patch('random.choices')
    # @patch('simulator.core.grid.SYMBOLS', new_callable=lambda: TestGridInitialization.DUMMY_SYMBOLS_DICT_BG)
    # @patch('simulator.core.grid.SYMBOL_GENERATION_WEIGHTS_BG', new_callable=lambda: TestGridInitialization.DUMMY_WEIGHTS_BG)
    # def test_initial_population_base_game(self, mock_weights, mock_symbols, mock_choices):
    def test_initial_population_base_game(self):
        """Test if the grid is populated correctly in Base Game mode."""
        self.state.is_free_spins = False

        # Simulate the initial spin population
        self.grid.initialize_spin()

        # Verify grid is populated (not EMPTY) and symbols are valid for BG
        valid_bg_symbols = set(SYMBOL_GENERATION_WEIGHTS_BG.keys())
        found_non_empty = False
        for r in range(GRID_ROWS):
            for c in range(GRID_COLS):
                symbol = self.grid.get_symbol(r, c)
                self.assertIsNotNone(symbol)
                self.assertNotEqual(symbol.name, "EMPTY")
                self.assertIn(symbol.name, valid_bg_symbols, f"Symbol {symbol.name} at ({r},{c}) not in BG weights")
                found_non_empty = True
        self.assertTrue(found_non_empty, "Grid was not populated with any symbols.")


    # @patch('random.choices')
    # @patch('simulator.core.grid.SYMBOLS', new_callable=lambda: TestGridInitialization.DUMMY_SYMBOLS_DICT_FS)
    # @patch('simulator.core.grid.SYMBOL_GENERATION_WEIGHTS_FS', new_callable=lambda: TestGridInitialization.DUMMY_WEIGHTS_FS)
    # def test_initial_population_free_spins(self, mock_weights, mock_symbols, mock_choices):
    def test_initial_population_free_spins(self):
        """Test if the grid is populated correctly in Free Spins mode."""
        self.state.is_free_spins = True

        # Simulate the initial spin population
        self.grid.initialize_spin()

        # Verify grid is populated (not EMPTY) and symbols are valid for FS
        valid_fs_symbols = set(SYMBOL_GENERATION_WEIGHTS_FS.keys())
        found_non_empty = False
        for r in range(GRID_ROWS):
            for c in range(GRID_COLS):
                symbol = self.grid.get_symbol(r, c)
                self.assertIsNotNone(symbol)
                self.assertNotEqual(symbol.name, "EMPTY")
                self.assertIn(symbol.name, valid_fs_symbols, f"Symbol {symbol.name} at ({r},{c}) not in FS weights")
                found_non_empty = True
        self.assertTrue(found_non_empty, "Grid was not populated with any symbols.")

# Ensure the script can be run directly
if __name__ == '__main__':
    unittest.main() 