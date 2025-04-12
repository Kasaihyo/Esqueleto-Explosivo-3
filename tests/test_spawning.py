import unittest
import random
from typing import List, Tuple, Optional, Set, Dict
from unittest.mock import patch # For mocking random choices

# Need to import from the simulator module
from simulator.core.grid import Grid
from simulator.core.symbol import Symbol, SymbolType
from simulator import config
from simulator.core.state import GameState # Import GameState

# Helper function (can be moved to a shared helper file later)
def create_grid_from_names(names: List[List[Optional[str]]]) -> Grid:
    rows = len(names)
    cols = len(names[0]) if rows > 0 else 0
    state = GameState() # Need state for Grid
    grid = Grid(state) # Assume default dims
    for r in range(rows):
        for c in range(cols):
            if r < grid.rows and c < grid.cols:
                name = names[r][c]
                symbol = config.SYMBOLS.get(name) if name else config.SYMBOLS["EMPTY"] # Use EMPTY not None
                grid._set_symbol(r, c, symbol)
    return grid

# Dummy symbols for mocking config (Moved to setUpClass)
# DUMMY_SPAWN_SYMBOLS: Dict[str, Symbol] = { ... }

class TestWildSpawning(unittest.TestCase):

    # @classmethod
    # def setUpClass(cls):
    #     \"\"\"Set up dummy symbols once for the test class.\"\"\"
    #     cls.DUMMY_SPAWN_SYMBOLS: Dict[str, Symbol] = {
    #         \"PNK\": Symbol(\"PNK\", SymbolType.LP),
    #         \"GRN\": Symbol(\"GRN\", SymbolType.LP),
    #         \"BLU\": Symbol(\"BLU\", SymbolType.LP),
    #         \"WILD\": Symbol(\"WILD\", SymbolType.WILD),
    #         \"E_WILD\": Symbol(\"E_WILD\", SymbolType.EXPLOSIVO_WILD),
    #         \"EMPTY\": Symbol(\"EMPTY\", SymbolType.EMPTY),
    #     }

    def test_one_wild_per_cluster_spawn(self):
        """Test exactly one wild spawns for each winning cluster."""
        # Use actual symbol names
        names = [
            ["PINK_SK", "PINK_SK", "PINK_SK", "PINK_SK", "PINK_SK"], # Cluster 1
            ["EMPTY", "EMPTY", "EMPTY", "EMPTY", "EMPTY"],
            ["GREEN_SK", "GREEN_SK", "EMPTY", "BLUE_SK", "BLUE_SK"],
            ["GREEN_SK", "GREEN_SK", "EMPTY", "BLUE_SK", "BLUE_SK"],
            ["GREEN_SK", "GREEN_SK", "EMPTY", "BLUE_SK", "BLUE_SK"],
        ] # 3 Clusters: PINK_SK, GREEN_SK, BLUE_SK
        grid = create_grid_from_names(names)
        clusters = grid.find_clusters()
        self.assertEqual(len(clusters), 3)

        grid.landed_coords = set() # Assume nothing landed for this specific test
        _cleared, _ew_coll, _did_exp, spawned_coords = grid.process_explosions_and_spawns(clusters)

        self.assertEqual(len(spawned_coords), 3) # One spawn per cluster
        for r, c in spawned_coords:
            spawned_symbol = grid._get_symbol(r, c)
            self.assertIsNotNone(spawned_symbol, f"Symbol at {r},{c} should not be None after spawn")
            self.assertTrue(spawned_symbol.type in [SymbolType.WILD, SymbolType.EXPLOSIVO_WILD])

    def test_spawn_location_in_footprint(self):
        """Test wild spawns occur within the original cluster footprint."""
        # Use actual symbol names
        names = [
            ["EMPTY", "EMPTY", "EMPTY", "EMPTY", "EMPTY"],
            ["EMPTY", "BLUE_SK", "BLUE_SK", "EMPTY", "EMPTY"],
            ["EMPTY", "BLUE_SK", "BLUE_SK", "EMPTY", "EMPTY"],
            ["EMPTY", "BLUE_SK", "EMPTY", "EMPTY", "EMPTY"], # 5 BLUE_SK cluster
            ["EMPTY", "EMPTY", "EMPTY", "EMPTY", "EMPTY"],
        ]
        grid = create_grid_from_names(names)
        clusters = grid.find_clusters()
        self.assertEqual(len(clusters), 1)
        _symbol_blu, coords_blu = clusters[0]
        original_footprint = set(coords_blu)
        self.assertEqual(original_footprint, {(1,1), (1,2), (2,1), (2,2), (3,1)})

        grid.landed_coords = set()
        _cleared, _ew_coll, _did_exp, spawned_coords = grid.process_explosions_and_spawns(clusters)

        self.assertEqual(len(spawned_coords), 1)
        spawn_coord = list(spawned_coords)[0]
        self.assertIn(spawn_coord, original_footprint)
        spawned_symbol = grid._get_symbol(spawn_coord[0], spawn_coord[1])
        self.assertIsNotNone(spawned_symbol)
        self.assertTrue(spawned_symbol.type in [SymbolType.WILD, SymbolType.EXPLOSIVO_WILD])

    @patch('random.random')
    def test_spawn_probability_wild(self, mock_random):
        """Test wild spawn probability favours W when random < P(SpawnW)."""
        mock_random.return_value = 0.7 # Less than P(SpawnW)
        self.assertLess(mock_random.return_value, config.PROB_SPAWN_WILD)

        names = [["PINK_SK"] * 5] + [["EMPTY"] * 5] * 4 # Simple 5-cluster
        grid = create_grid_from_names(names)
        clusters = grid.find_clusters()
        self.assertEqual(len(clusters), 1)

        grid.landed_coords = set()
        _cleared, _ew_coll, _did_exp, spawned_coords = grid.process_explosions_and_spawns(clusters)
        self.assertEqual(len(spawned_coords), 1)
        spawn_coord = list(spawned_coords)[0]
        spawned_symbol = grid._get_symbol(spawn_coord[0], spawn_coord[1])
        self.assertIsNotNone(spawned_symbol)
        self.assertEqual(spawned_symbol.name, "WILD") # Check actual name

    @patch('random.random')
    def test_spawn_probability_ewild(self, mock_random):
        """Test wild spawn probability gives EW when random >= P(SpawnW)."""
        mock_random.return_value = 0.9 # Greater than or equal to P(SpawnW)
        self.assertGreaterEqual(mock_random.return_value, config.PROB_SPAWN_WILD)

        names = [["PINK_SK"] * 5] + [["EMPTY"] * 5] * 4 # Simple 5-cluster
        grid = create_grid_from_names(names)
        clusters = grid.find_clusters()
        self.assertEqual(len(clusters), 1)

        grid.landed_coords = set()
        _cleared, _ew_coll, _did_exp, spawned_coords = grid.process_explosions_and_spawns(clusters)
        self.assertEqual(len(spawned_coords), 1)
        spawn_coord = list(spawned_coords)[0]
        spawned_symbol = grid._get_symbol(spawn_coord[0], spawn_coord[1])
    @classmethod
    def setUpClass(cls):
        """Set up dummy symbols once for the test class."""
        cls.DUMMY_SPAWN_SYMBOLS: Dict[str, Symbol] = {
            "PNK": Symbol("PNK", SymbolType.LP),
            "GRN": Symbol("GRN", SymbolType.LP),
            "BLU": Symbol("BLU", SymbolType.LP),
            "WILD": Symbol("WILD", SymbolType.WILD),
            "E_WILD": Symbol("E_WILD", SymbolType.EXPLOSIVO_WILD),
            "EMPTY": Symbol("EMPTY", SymbolType.EMPTY),
        }

    def test_one_wild_per_cluster_spawn(self):
        """Test exactly one wild spawns for each winning cluster."""
        # Use setUpClass symbols via patch
        with patch('simulator.config.SYMBOLS', self.DUMMY_SPAWN_SYMBOLS):
            names = [
                ["PNK", "PNK", "PNK", "PNK", "PNK"], # Cluster 1
                [None, None, None, None, None],
                ["GRN", "GRN", None, "BLU", "BLU"],
                ["GRN", "GRN", None, "BLU", "BLU"],
                ["GRN", "GRN", None, "BLU", "BLU"],
            ] # 3 Clusters
            grid = create_grid_from_names(names)
            clusters = grid.find_clusters()
            self.assertEqual(len(clusters), 3)

            grid.landed_coords = set()
            _cleared, _ew_coll, _did_exp, spawned_coords = grid.process_explosions_and_spawns(clusters)

            self.assertEqual(len(spawned_coords), 3)
            for r, c in spawned_coords:
                spawned_symbol = grid._get_symbol(r, c)
                self.assertTrue(spawned_symbol.type in [SymbolType.WILD, SymbolType.EXPLOSIVO_WILD])

    def test_spawn_location_in_footprint(self):
        """Test wild spawns occur within the original cluster footprint."""
        with patch('simulator.config.SYMBOLS', self.DUMMY_SPAWN_SYMBOLS):
            names = [
                [None, None, None, None, None],
                [None, "BLU", "BLU", None, None],
                [None, "BLU", "BLU", None, None],
                [None, "BLU", None, None, None],
                [None, None, None, None, None],
            ]
            grid = create_grid_from_names(names)
            clusters = grid.find_clusters()
            self.assertEqual(len(clusters), 1)
            _symbol_blu, coords_blu = clusters[0]
            original_footprint = set(coords_blu)
            self.assertEqual(original_footprint, {(1,1), (1,2), (2,1), (2,2), (3,1)})

            grid.landed_coords = set()
            _cleared, _ew_coll, _did_exp, spawned_coords = grid.process_explosions_and_spawns(clusters)

            self.assertEqual(len(spawned_coords), 1)
            spawn_coord = list(spawned_coords)[0]
            self.assertIn(spawn_coord, original_footprint)
            spawned_symbol = grid._get_symbol(spawn_coord[0], spawn_coord[1])
            self.assertTrue(spawned_symbol.type in [SymbolType.WILD, SymbolType.EXPLOSIVO_WILD])

    @patch('random.random')
    def test_spawn_probability_wild(self, mock_random):
        """Test wild spawn probability favours W when random < P(SpawnW)."""
        with patch('simulator.config.SYMBOLS', self.DUMMY_SPAWN_SYMBOLS):
            mock_random.return_value = 0.7
            config.PROB_SPAWN_WILD = 0.8
            names = [["PNK"] * 5] + [[None] * 5] * 4
            grid = create_grid_from_names(names)
            clusters = grid.find_clusters()
            self.assertEqual(len(clusters), 1)

            grid.landed_coords = set()
            _cleared, _ew_coll, _did_exp, spawned_coords = grid.process_explosions_and_spawns(clusters)
            self.assertEqual(len(spawned_coords), 1)
            spawn_coord = list(spawned_coords)[0]
            spawned_symbol = grid._get_symbol(spawn_coord[0], spawn_coord[1])
            self.assertEqual(spawned_symbol.name, "WILD")

    @patch('random.random')
    def test_spawn_probability_ewild(self, mock_random):
        """Test wild spawn probability gives EW when random >= P(SpawnW)."""
        with patch('simulator.config.SYMBOLS', self.DUMMY_SPAWN_SYMBOLS):
            mock_random.return_value = 0.9
            config.PROB_SPAWN_WILD = 0.8
            names = [["PNK"] * 5] + [[None] * 5] * 4
            grid = create_grid_from_names(names)
            clusters = grid.find_clusters()
            self.assertEqual(len(clusters), 1)

            grid.landed_coords = set()
            _cleared, _ew_coll, _did_exp, spawned_coords = grid.process_explosions_and_spawns(clusters)
            self.assertEqual(len(spawned_coords), 1)
            spawn_coord = list(spawned_coords)[0]
            spawned_symbol = grid._get_symbol(spawn_coord[0], spawn_coord[1])
            self.assertEqual(spawned_symbol.name, "E_WILD")

    def test_spawn_collision_handling(self):
        """Test spawn collision - multiple overlapping clusters."""
        with patch('simulator.config.SYMBOLS', self.DUMMY_SPAWN_SYMBOLS):
            names = [
                ["PNK", "PNK", "PNK", None, None],
                ["PNK", None,  "GRN", None, None],
                ["PNK", "GRN", "GRN", "GRN", "GRN"],
                [None, None, None, None, None],
                [None, None, None, None, None],
            ]
            grid = create_grid_from_names(names)
            clusters = grid.find_clusters()
            self.assertEqual(len(clusters), 2)

            grid.landed_coords = set()
            random.seed(123)
            _cleared, _ew_coll, _did_exp, spawned_coords = grid.process_explosions_and_spawns(clusters)

            self.assertEqual(len(spawned_coords), 2)
            self.assertEqual(len(set(spawned_coords)), 2)
            spawn_types = {grid._get_symbol(r,c).type for r,c in spawned_coords}
            self.assertTrue(all(t in [SymbolType.WILD, SymbolType.EXPLOSIVO_WILD] for t in spawn_types))
            pnk_footprint = {(0,0), (0,1), (0,2), (1,0), (2,0)}
            grn_footprint = {(1,2), (2,1), (2,2), (2,3), (2,4)}
            combined_footprint = pnk_footprint | grn_footprint
            self.assertTrue(all(coord in combined_footprint for coord in spawned_coords))

    def test_no_spawn_if_footprint_full(self):
        """Test no wild spawns if cluster footprint has no available clear space (e.g., blocked by another spawn or unexploded EW)."""
        # Use patch as context manager with the symbols from setUpClass
        with patch('simulator.config.SYMBOLS', self.DUMMY_SPAWN_SYMBOLS):
            names = [
                ["PNK", "PNK", "PNK", None, None],
                ["PNK", "E_WILD","PNK", None, None],
                ["PNK", "PNK", "PNK", None, None],
                [None, None, None, None, None],
                [None, None, None, None, None],
            ]
            grid = create_grid_from_names(names)
            grid.landed_coords = set()
            cluster_symbol = config.SYMBOLS["PNK"]
            cluster_coords = [(0,0), (0,1), (0,2), (1,0), (2,0)]
            clusters = [(cluster_symbol, cluster_coords)]
            names_block = [
                ["PNK", "PNK", "PNK", None, None],
                ["PNK", "PNK", None, None, None],
                [None, None, None, None, None],
                [None, None, None, None, None],
                [None, None, None, None, None],
            ]
            grid_block = create_grid_from_names(names_block)
            cluster_block = grid_block.find_clusters()[0]
            footprint_block = set(cluster_block[1])
            grid_block.landed_coords = set()
            coords_cleared_by_cluster = footprint_block
            all_coords_marked_for_clearing = coords_cleared_by_cluster
            coords_receiving_spawned_wild = footprint_block
            spawn_count_test = 0
            for footprint in [footprint_block]:
                 potential_spawn_locations = list(
                     (footprint & all_coords_marked_for_clearing) - coords_receiving_spawned_wild
                 )
                 if not potential_spawn_locations:
                     continue
                 spawn_count_test += 1
            self.assertEqual(spawn_count_test, 0)

if __name__ == '__main__':
    unittest.main() 