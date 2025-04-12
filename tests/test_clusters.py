import unittest
from typing import List, Tuple, Optional, Set

# Need to import from the simulator module (assuming run from project root)
from simulator.core.grid import Grid
from simulator.core.symbol import Symbol, SymbolType
from simulator import config
from simulator.core.state import GameState # Import GameState

# Helper function to create a grid from a list of lists of symbol names
def create_grid_from_names(names: List[List[Optional[str]]]) -> Grid:
    rows = len(names)
    cols = len(names[0]) if rows > 0 else 0
    state = GameState() # Grid requires state
    grid = Grid(state) # Assume Grid gets dimensions from config or state?
    # We need to ensure this grid has the right dimensions for the test
    # If Grid doesn't allow custom dims anymore, we might need to mock config
    # For now, assume 5x5 is default and test data fits.
    # OR, perhaps Grid init should be: Grid(state, rows=rows, cols=cols) if state is NOT passed?
    # Let's assume for now the constructor is Grid(state) and uses default dims.
    # If tests fail later due to size, we'll revisit.
    for r in range(rows):
        for c in range(cols):
            # Add boundary checks if grid uses default size
            if r < grid.rows and c < grid.cols:
                name = names[r][c]
                symbol = config.SYMBOLS.get(name) if name else config.SYMBOLS["EMPTY"]
                grid._set_symbol(r, c, symbol)
            # else: handle error or ignore if test assumes matching dims
    return grid

class TestClusterDetection(unittest.TestCase):

    def test_cluster_detection_simple_horizontal(self):
        """Test finding a simple horizontal cluster."""
        # Use actual symbol names
        names = [
            ["EMPTY", "EMPTY", "EMPTY", "EMPTY", "EMPTY"],
            ["EMPTY", "EMPTY", "EMPTY", "EMPTY", "EMPTY"],
            ["PINK_SK", "PINK_SK", "PINK_SK", "PINK_SK", "PINK_SK"],
            ["EMPTY", "EMPTY", "EMPTY", "EMPTY", "EMPTY"],
            ["EMPTY", "EMPTY", "EMPTY", "EMPTY", "EMPTY"],
        ]
        grid = create_grid_from_names(names)
        clusters = grid.find_clusters()
        self.assertEqual(len(clusters), 1)
        symbol, coords = clusters[0]
        self.assertEqual(symbol.name, "PINK_SK") # Check correct name
        self.assertEqual(len(coords), 5)
        self.assertSetEqual(set(coords), {(2, 0), (2, 1), (2, 2), (2, 3), (2, 4)})

    def test_cluster_detection_simple_vertical(self):
        """Test finding a simple vertical cluster."""
        # Use actual symbol names
        names = [
            ["EMPTY", "GREEN_SK", "EMPTY", "EMPTY", "EMPTY"],
            ["EMPTY", "GREEN_SK", "EMPTY", "EMPTY", "EMPTY"],
            ["EMPTY", "GREEN_SK", "EMPTY", "EMPTY", "EMPTY"],
            ["EMPTY", "GREEN_SK", "EMPTY", "EMPTY", "EMPTY"],
            ["EMPTY", "GREEN_SK", "EMPTY", "EMPTY", "EMPTY"],
        ]
        grid = create_grid_from_names(names)
        clusters = grid.find_clusters()
        self.assertEqual(len(clusters), 1)
        symbol, coords = clusters[0]
        self.assertEqual(symbol.name, "GREEN_SK") # Check correct name
        self.assertEqual(len(coords), 5)
        self.assertSetEqual(set(coords), {(0, 1), (1, 1), (2, 1), (3, 1), (4, 1)})

    def test_cluster_detection_l_shape(self):
        """Test finding an L-shaped cluster."""
        # Use actual symbol names
        names = [
            ["EMPTY", "EMPTY", "EMPTY", "EMPTY", "EMPTY"],
            ["EMPTY", "BLUE_SK", "EMPTY", "EMPTY", "EMPTY"],
            ["EMPTY", "BLUE_SK", "EMPTY", "EMPTY", "EMPTY"],
            ["EMPTY", "BLUE_SK", "BLUE_SK", "BLUE_SK", "EMPTY"],
            ["EMPTY", "EMPTY", "EMPTY", "EMPTY", "EMPTY"],
        ]
        grid = create_grid_from_names(names)
        clusters = grid.find_clusters()
        self.assertEqual(len(clusters), 1)
        symbol, coords = clusters[0]
        self.assertEqual(symbol.name, "BLUE_SK") # Check correct name
        self.assertEqual(len(coords), 5)
        self.assertSetEqual(set(coords), {(1, 1), (2, 1), (3, 1), (3, 2), (3, 3)})

    def test_cluster_detection_with_wilds(self):
        """Test finding a cluster connected by wilds."""
        # Use actual symbol names
        names = [
            ["EMPTY", "EMPTY", "EMPTY", "EMPTY", "EMPTY"],
            ["EMPTY", "ORANGE_SK", "WILD", "ORANGE_SK", "EMPTY"],
            ["EMPTY", "EMPTY", "WILD", "EMPTY", "EMPTY"],
            ["EMPTY", "EMPTY", "ORANGE_SK", "ORANGE_SK", "ORANGE_SK"],
            ["EMPTY", "EMPTY", "EMPTY", "EMPTY", "EMPTY"],
        ]
        grid = create_grid_from_names(names)
        clusters = grid.find_clusters()
        self.assertEqual(len(clusters), 1)
        symbol, coords = clusters[0]
        self.assertEqual(symbol.name, "ORANGE_SK") # Should identify as ORANGE_SK cluster
        self.assertEqual(len(coords), 7) # 5 ORG + 2 WILD
        self.assertSetEqual(set(coords), {
            (1, 1), (1, 2), (1, 3), # Top row ORG, WILD, ORG
            (2, 2),                 # Middle WILD
            (3, 2), (3, 3), (3, 4)  # Bottom row ORG, ORG, ORG
        })

    def test_cluster_detection_no_cluster(self):
        """Test grid with no winning clusters."""
        # Use actual symbol names
        names = [
            ["PINK_SK", "GREEN_SK", "BLUE_SK", "ORANGE_SK", "CYAN_SK"],
            ["LADY_SK", "PINK_SK", "GREEN_SK", "BLUE_SK", "ORANGE_SK"],
            ["CYAN_SK", "LADY_SK", "PINK_SK", "GREEN_SK", "BLUE_SK"],
            ["ORANGE_SK", "CYAN_SK", "LADY_SK", "PINK_SK", "GREEN_SK"],
            ["BLUE_SK", "ORANGE_SK", "CYAN_SK", "LADY_SK", "PINK_SK"],
        ]
        grid = create_grid_from_names(names)
        clusters = grid.find_clusters()
        self.assertEqual(len(clusters), 0)

    def test_cluster_detection_multiple_clusters(self):
        """Test grid with multiple separate winning clusters."""
        # Use actual symbol names
        names = [
            ["PINK_SK", "PINK_SK", "PINK_SK", "PINK_SK", "PINK_SK"],
            ["EMPTY", "EMPTY", "EMPTY", "EMPTY", "EMPTY"],
            ["GREEN_SK", "GREEN_SK", "EMPTY", "BLUE_SK", "BLUE_SK"],
            ["GREEN_SK", "GREEN_SK", "EMPTY", "BLUE_SK", "BLUE_SK"],
            ["GREEN_SK", "GREEN_SK", "EMPTY", "BLUE_SK", "BLUE_SK"],
        ]
        grid = create_grid_from_names(names)
        clusters = grid.find_clusters()
        self.assertEqual(len(clusters), 3)
        # Sort clusters by symbol name for consistent checking
        clusters.sort(key=lambda x: x[0].name)

        # Check Blue cluster
        symbol_blu, coords_blu = clusters[0]
        self.assertEqual(symbol_blu.name, "BLUE_SK") # Check correct name
        self.assertEqual(len(coords_blu), 6)
        self.assertSetEqual(set(coords_blu), {(2, 3), (2, 4), (3, 3), (3, 4), (4, 3), (4, 4)})

        # Check Green cluster
        symbol_grn, coords_grn = clusters[1]
        self.assertEqual(symbol_grn.name, "GREEN_SK") # Check correct name
        self.assertEqual(len(coords_grn), 6)
        self.assertSetEqual(set(coords_grn), {(2, 0), (2, 1), (3, 0), (3, 1), (4, 0), (4, 1)})

        # Check Pink cluster
        symbol_pnk, coords_pnk = clusters[2]
        self.assertEqual(symbol_pnk.name, "PINK_SK") # Check correct name
        self.assertEqual(len(coords_pnk), 5)
        self.assertSetEqual(set(coords_pnk), {(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)})

    def test_cluster_detection_size_4_ignored(self):
        """Test that a cluster of size 4 is not detected (min size 5)."""
        # Use actual symbol names
        names = [
            ["EMPTY", "EMPTY", "EMPTY", "EMPTY", "EMPTY"],
            ["EMPTY", "BLUE_SK", "BLUE_SK", "EMPTY", "EMPTY"],
            ["EMPTY", "BLUE_SK", "BLUE_SK", "EMPTY", "EMPTY"],
            ["EMPTY", "EMPTY", "EMPTY", "EMPTY", "EMPTY"],
            ["EMPTY", "EMPTY", "EMPTY", "EMPTY", "EMPTY"],
        ]
        grid = create_grid_from_names(names)
        clusters = grid.find_clusters()
        self.assertEqual(len(clusters), 0)

    def test_cluster_detection_non_paying_symbols(self):
        """Test that non-paying symbols (W, EW, S, EMPTY) do not form clusters."""
        # Use actual symbol names
        names = [
            ["WILD", "WILD", "WILD", "WILD", "WILD"],
            ["SCATTER", "SCATTER", "SCATTER", "SCATTER", "SCATTER"],
            ["E_WILD", "E_WILD", "E_WILD", "E_WILD", "E_WILD"],
            ["EMPTY", "EMPTY", "EMPTY", "EMPTY", "EMPTY"], # Representing EMPTY
            ["WILD", "SCATTER", "E_WILD", "EMPTY", "WILD"],
        ]
        grid = create_grid_from_names(names)
        clusters = grid.find_clusters()
        self.assertEqual(len(clusters), 0)

if __name__ == '__main__':
    unittest.main() 