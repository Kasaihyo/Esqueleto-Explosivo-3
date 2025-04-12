import unittest
from typing import List, Tuple, Optional, Set

# Need to import from the simulator module
from simulator.core.grid import Grid
from simulator.core.symbol import Symbol, SymbolType
from simulator import config
from simulator.core.state import GameState # Import GameState

# Helper function (can be moved to a shared helper file later)
# --- REMOVE UNUSED HELPER --- #
# def create_grid_from_names(names: List[List[Optional[str]]]) -> Grid:
#     rows = len(names)
#     cols = len(names[0]) if rows > 0 else 0
#     grid = Grid(rows=rows, cols=cols)
#     for r in range(rows):
#         for c in range(cols):
#             name = names[r][c]
#             symbol = config.SYMBOLS.get(name) if name else None
#             grid._set_symbol(r, c, symbol)
#     return grid

class TestPayouts(unittest.TestCase):

    def setUp(self): # Use setUp to create a grid instance for tests if needed
        self.state = GameState()
        # This init caused TypeError: 'GameState' object cannot be interpreted as an integer
        # This implies Grid(state) IS the correct call, but the Grid.__init__ itself has an issue
        # Let's assume the Grid.__init__ needs fixing externally.
        # The call here seems correct based on other errors.
        self.grid = Grid(self.state) # Standard 5x5 grid, requires state
        self.base_bet = config.BASE_BET # Use base bet from config

    def test_payout_min_cluster_lp(self):
        """Test payout for minimum size (5) low-pay cluster."""
        symbol = config.SYMBOLS["CYAN_SK"] # Lowest LP
        coords = [(0,0), (0,1), (0,2), (1,0), (1,1)]
        clusters = [(symbol, coords)]
        multiplier = 1 # Base game, first avalanche
        expected_payout = self.base_bet * config.PAYTABLE["CYAN_SK"][5] * multiplier
        actual_payout = self.grid.calculate_win(clusters, self.base_bet, multiplier)
        self.assertAlmostEqual(actual_payout, expected_payout)

    def test_payout_mid_cluster_hp(self):
        """Test payout for medium size (9) high-pay cluster."""
        symbol = config.SYMBOLS["LADY_SK"] # HP
        coords = [(0,0), (0,1), (0,2), (1,0), (1,1), (1,2), (2,0), (2,1), (2,2)] # 9 symbols
        clusters = [(symbol, coords)]
        multiplier = 1
        # Payout uses the 8-9 bracket (index 9 in config)
        expected_payout = self.base_bet * config.PAYTABLE["LADY_SK"][9] * multiplier
        actual_payout = self.grid.calculate_win(clusters, self.base_bet, multiplier)
        self.assertAlmostEqual(actual_payout, expected_payout)

    def test_payout_max_cluster_size(self):
        """Test payout uses max defined size (15+) for larger clusters."""
        symbol = config.SYMBOLS["PINK_SK"]
        # Create a list of 16 coordinates
        coords = [(r, c) for r in range(4) for c in range(4)]
        self.assertEqual(len(coords), 16)
        clusters = [(symbol, coords)]
        multiplier = 1
        # Payout should use the value for cluster size 15 (index 15 in config)
        expected_payout = self.base_bet * config.PAYTABLE["PINK_SK"][15] * multiplier
        actual_payout = self.grid.calculate_win(clusters, self.base_bet, multiplier)
        self.assertAlmostEqual(actual_payout, expected_payout)

    def test_payout_base_game_multiplier(self):
        """Test payout with a base game multiplier."""
        symbol = config.SYMBOLS["GREEN_SK"]
        coords = [(0,0), (0,1), (0,2), (0,3), (0,4)] # 5 symbols
        clusters = [(symbol, coords)]
        multiplier = 8 # Example BG multiplier
        expected_payout = self.base_bet * config.PAYTABLE["GREEN_SK"][5] * multiplier
        actual_payout = self.grid.calculate_win(clusters, self.base_bet, multiplier)
        self.assertAlmostEqual(actual_payout, expected_payout)

    def test_payout_free_spins_multiplier(self):
        """Test payout logic with an FS multiplier (using config rules)."""
        # Simulate FS state: Base Level 4x, 3rd avalanche (index 2)
        fs_base = 4
        fs_avalanche_count = 2 # Corresponds to index 2 in the trail

        # Calculate multiplier based on config.FS_MUCHO_MULTIPLIER_TRAIL
        # Find the trail for the base multiplier
        trail = config.FS_MUCHO_MULTIPLIER_TRAIL.get(fs_base)
        self.assertIsNotNone(trail, f"FS Multiplier trail not found for base {fs_base}")
        # Get the multiplier at the specified avalanche index (capped at trail length)
        multiplier_index = min(fs_avalanche_count, len(trail) - 1)
        multiplier = trail[multiplier_index]

        # From config: Base 4x -> Trail: 4x, 8x, 16x, 32x, 64x, 128x. Index 2 should be 16x
        self.assertEqual(multiplier, 16)

        symbol = config.SYMBOLS["LADY_SK"]
        coords = [(0,0), (0,1), (1,0), (1,1), (2,0)] # 5 symbols
        clusters = [(symbol, coords)]
        expected_payout = self.base_bet * config.PAYTABLE["LADY_SK"][5] * multiplier
        actual_payout = self.grid.calculate_win(clusters, self.base_bet, multiplier)
        self.assertAlmostEqual(actual_payout, expected_payout)

    def test_payout_zero_for_non_paying(self):
        """Test zero payout for clusters of non-paying symbols."""
        clusters = [
            (config.SYMBOLS["WILD"], [(0,0), (0,1), (0,2), (0,3), (0,4)]),
            (config.SYMBOLS["SCATTER"], [(1,0), (1,1), (1,2), (1,3), (1,4)]),
            (config.SYMBOLS["E_WILD"], [(2,0), (2,1), (2,2), (2,3), (2,4)]),
        ]
        multiplier = 1
        actual_payout = self.grid.calculate_win(clusters, self.base_bet, multiplier)
        self.assertAlmostEqual(actual_payout, 0.0)

    def test_payout_no_clusters(self):
        """Test zero payout when no clusters are provided."""
        clusters = []
        multiplier = 1
        actual_payout = self.grid.calculate_win(clusters, self.base_bet, multiplier)
        self.assertAlmostEqual(actual_payout, 0.0)


if __name__ == '__main__':
    unittest.main() 