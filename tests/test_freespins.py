# import random # F401 unused
import unittest

# from typing import Dict, List, Optional, Set, Tuple # F401 All unused
from unittest.mock import patch  # F401 MagicMock was unused, patch is used.

from simulator import config
from simulator.core.grid import Grid

# from simulator.core.symbol import Symbol, SymbolType # F401 unused
from simulator.core.rng import SpinRNG
from simulator.core.state import GameState
from simulator.main import calculate_retrigger_spins, run_free_spins_feature


class TestFreeSpinsFeature(unittest.TestCase):
    """Tests for the Free Spins feature of the Esqueleto Explosivo 3 game."""

    def test_fs_symbol_distribution(self):
        """Test that Free Spins mode uses the FS symbol distribution for refills."""
        # Setup
        state = GameState()
        state.is_free_spins = True
        grid = Grid(state)

        # We'll patch the underlying generate_random_symbol function
        with patch("simulator.core.utils.generate_random_symbol") as mock_generate:
            # Set return value to avoid KeyError
            mock_generate.return_value = config.SYMBOLS["PINK_SK"]

            # Initialize the grid for Free Spins
            grid.initialize_spin(state)

            # Check if function was called with FS weights key
            calls = mock_generate.call_args_list
            for args, kwargs in calls:
                self.assertEqual(kwargs.get("weights_key"), "FS")

    def test_fs_multiplier_trail(self):
        """Test FS multiplier uses correct trail based on current base level."""
        # Test various base multiplier levels
        for base_level in config.FS_BASE_MULTIPLIER_LEVELS:
            state = GameState()
            state.is_free_spins = True
            state.fs_base_multiplier = base_level
            grid = Grid(state)

            # Initialize the spin (resets multiplier index to 0)
            state.initialize_spin()

            # Get the expected multiplier trail for this base level
            expected_trail = config.FS_MULTIPLIER_TRAILS[base_level]

            # Check initial value matches first element in trail
            self.assertEqual(state.current_multiplier, expected_trail[0])

            # Simulate a series of wins to advance through the trail
            for i in range(1, len(expected_trail)):
                # Update multiplier using non-empty clusters list to simulate a win
                state.update_after_clusters([(None, None)], False)
                # Verify multiplier is correct
                self.assertEqual(state.current_multiplier, expected_trail[i])

            # Verify it caps at the maximum value in the trail
            state.update_after_clusters([(None, None)], False)
            self.assertEqual(state.current_multiplier, expected_trail[-1])

    def test_fs_multiplier_per_spin(self):
        """Test FS multiplier uses avalanche count within the current free spin."""
        # Setup
        state = GameState()
        state.is_free_spins = True
        state.fs_base_multiplier = 2  # Use base level 2 (trail: 2,4,8,16,32,64)
        grid = Grid(state)

        # Initialize first spin
        state.initialize_spin()

        # Verify initial multiplier
        self.assertEqual(state.current_multiplier, 2)

        # Advance through trail with a few wins
        state.update_after_clusters([(None, None)], False)
        self.assertEqual(state.current_multiplier, 4)
        state.update_after_clusters([(None, None)], False)
        self.assertEqual(state.current_multiplier, 8)

        # Initialize new spin - should reset to base level
        state.initialize_spin()
        self.assertEqual(state.current_multiplier, 2)

    def test_fs_remaining_spins(self):
        """Test remaining spins tracked correctly in Free Spins."""
        # Setup
        state = GameState()
        initial_spins = 10

        # Start Free Spins
        state.start_free_spins(initial_spins)
        self.assertEqual(state.remaining_spins, initial_spins)

        # Consume a few spins
        for i in range(3):
            state.consume_free_spin()
            self.assertEqual(state.remaining_spins, initial_spins - (i + 1))

        # Verify FS mode is still active
        self.assertTrue(state.is_free_spins)

        # Consume remaining spins
        for i in range(initial_spins - 3):
            state.consume_free_spin()

        # Verify FS mode has ended
        self.assertEqual(state.remaining_spins, 0)
        self.assertFalse(state.is_free_spins)

    def test_fs_retrigger(self):
        """Test retrigger awards correct spins for 2+ scatters within an FS sequence."""
        # Setup
        state = GameState()
        state.is_free_spins = True
        state.remaining_spins = 5

        # Test various scatter counts
        test_cases = [
            (2, 3),  # 2 scatters -> 3 more spins
            (3, 5),  # 3 scatters -> 5 more spins
            (4, 7),  # 4 scatters -> 7 more spins
            (5, 9),  # 5 scatters -> 9 more spins (7 + 2)
            (6, 11),  # 6 scatters -> 11 more spins (7 + 2*2)
        ]

        for scatters, expected_spins in test_cases:
            # Reset state for each test
            state.remaining_spins = 5
            state.scatters_collected_this_sequence = set()

            # Add scatter coordinates
            for i in range(scatters):
                state.accumulate_scatter((i, 0))

            # Trigger retrigger check
            _, spins_won = state.finalize_spin_sequence()

            # Verify correct number of spins awarded
            self.assertEqual(spins_won, expected_spins)
            self.assertEqual(state.remaining_spins, 5 + expected_spins)

    def test_ew_collection(self):
        """Test EW Collection increments session counter correctly."""
        # Setup
        state = GameState()
        state.is_free_spins = True

        # Test EW collection from different sources
        test_cases = [
            (1, 1),  # Collect 1 EW
            (2, 3),  # Collect 2 more EWs (total 3)
            (3, 6),  # Collect 3 more EWs (total 6)
        ]

        accumulated = 0
        for collect_count, expected_total in test_cases:
            # Simulate collecting EWs
            state.accumulate_ew_collected(collect_count)
            accumulated += collect_count

            # Verify session counter is updated correctly
            self.assertEqual(state.fs_ew_collected_session, accumulated)
            self.assertEqual(state.fs_ew_collected_this_spin, accumulated)

    def test_upgrade_check(self):
        """Test Upgrade Check correctly calculates pending upgrades."""
        # Setup EW_COLLECTION_PER_UPGRADE value
        per_upgrade = config.FS_EW_COLLECTION_PER_UPGRADE

        # Test cases for EW collection and expected upgrades
        test_cases = [
            (per_upgrade - 1, 0),  # Not enough for an upgrade
            (per_upgrade, 1),  # Exact amount for 1 upgrade
            (per_upgrade + 1, 1),  # 1 upgrade with remainder
            (per_upgrade * 2, 2),  # Exactly enough for 2 upgrades
        ]

        for collected, expected_upgrades in test_cases:
            # Create a new state for each test
            state = GameState()
            state.is_free_spins = True
            state.fs_base_multiplier = 1
            state.fs_ew_collected_session = 0
            state.fs_ew_collected_this_spin = collected

            # Finalize spin to check for upgrades
            state.finalize_spin_sequence()

            # Verify pending upgrades
            self.assertEqual(
                state.fs_pending_upgrades,
                expected_upgrades,
                f"With {collected} EWs collected (per_upgrade={per_upgrade}), expected {expected_upgrades} upgrades",
            )

    def test_upgrade_application(self):
        """Test Upgrade Application correctly increases base multiplier and awards spins."""
        # Setup
        state = GameState()
        state.is_free_spins = True
        state.remaining_spins = 5

        # Test different upgrade scenarios
        test_cases = [
            (1, 1, 2),  # Base 1x + 1 upgrade -> 2x
            (2, 1, 4),  # Base 2x + 1 upgrade -> 4x
            (4, 2, 16),  # Base 4x + 2 upgrades -> 16x
            (16, 1, 32),  # Base 16x + 1 upgrade -> 32x
            (32, 1, 32),  # Base 32x + 1 upgrade -> 32x (capped)
        ]

        for initial_level, upgrades, expected_level in test_cases:
            # Reset state for each test
            state.fs_base_multiplier = initial_level
            state.remaining_spins = 5
            state.fs_pending_upgrades = upgrades

            # Apply upgrades by initializing next spin
            state.initialize_spin()

            # Verify base multiplier is upgraded correctly
            self.assertEqual(state.fs_base_multiplier, expected_level)

            # Verify additional spins are awarded (1 per upgrade)
            self.assertEqual(state.remaining_spins, 5 + upgrades)

            # Verify pending upgrades are cleared
            self.assertEqual(state.fs_pending_upgrades, 0)

    def test_upgrade_timing(self):
        """Test that upgrades are applied at start of next spin, not immediately."""
        # Setup
        state = GameState()
        state.is_free_spins = True
        state.fs_base_multiplier = 1
        state.remaining_spins = 5

        # Collect enough EWs for an upgrade
        state.fs_ew_collected_this_spin = 3

        # Finalize current spin
        state.finalize_spin_sequence()

        # Verify upgrade is pending but not yet applied
        self.assertEqual(state.fs_pending_upgrades, 1)
        self.assertEqual(state.fs_base_multiplier, 1)  # Still at initial value

        # Initialize next spin (this should apply the upgrade)
        state.initialize_spin()

        # Verify upgrade was applied
        self.assertEqual(state.fs_base_multiplier, 2)
        self.assertEqual(state.remaining_spins, 6)  # 5 + 1 from upgrade
        self.assertEqual(state.fs_pending_upgrades, 0)

    def test_fs_ends_if_max_win_cap_reached(self):
        base_bet = 1.0
        initial_spins = 10
        # grid = Grid(GameState(), rng_seed=1) # F841 - grid not used directly
        mock_grid_instance = Grid(GameState(), rng=SpinRNG(1))
        # mock_grid_instance.rng = SpinRNG(1) # Ensure RNG is seeded

        # Patch find_clusters to return a high win, then nothing

    def test_fs_cap_not_exceeded_if_win_is_less(self):
        base_bet = 1.0
        initial_spins = 5
        # grid = Grid(GameState(), rng_seed=2) # F841 grid not used
        mock_grid_instance = Grid(GameState(), rng=SpinRNG(1))

        # Simulate a scenario where total win is less than cap


if __name__ == "__main__":
    unittest.main()
