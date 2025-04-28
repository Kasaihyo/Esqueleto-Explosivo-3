import pytest
from simulator.main import run_free_spins_feature

class DummyGrid:
    def __init__(self, scatters_sequence):
        self.scatters_sequence = list(scatters_sequence)
        self.initialized_count = 0

    def initialize_spin(self, game_state):
        # Count Free Spin starts
        self.initialized_count += 1

    def count_scatters(self):
        # Return next scatter count or 0
        return self.scatters_sequence.pop(0) if self.scatters_sequence else 0

    def find_clusters(self):
        return []  # No clusters/wins

    def process_explosions_and_spawns(self, clusters):
        # No explosions or spawns
        return set(), 0, False, set()

    def apply_avalanche(self, game_state):
        # No symbols land, immediate end
        return set()

@ pytest.mark.parametrize("initial_spins, scatters_sequence, expected_spins", [
    (3, [0, 0, 0], 3),         # No retriggers, should use exactly initial spins
    (1, [2], 1 + 3),           # One retrigger for scatter_count=2 (3 extra spins)
    (1, [3], 1 + 5),           # One retrigger for scatter_count=3 (5 extra spins)
])
def test_run_free_spins_retrigger_counts(initial_spins, scatters_sequence, expected_spins):
    """Test that run_free_spins_feature processes the correct number of spins based on retriggers."""
    grid = DummyGrid(scatters_sequence)
    # Verbose and base bet args not used for retrigger count
    _ = run_free_spins_feature(grid, base_bet=1.0, initial_spins=initial_spins, trigger_spin_index=0, verbose=False)
    assert grid.initialized_count == expected_spins 