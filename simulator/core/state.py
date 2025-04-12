# simulator/core/state.py

from typing import List, Tuple, Optional, Set, Dict
from simulator import config # Import config for multiplier/spin rules

class GameState:
    """Manages the state of a single spin sequence (Base Game or Free Spins)."""

    def __init__(self):
        # Core state attributes
        self.is_free_spins: bool = False
        self.remaining_spins: int = 0 # For FS
        self.current_multiplier: int = 1
        self.current_multiplier_index: int = 0 # Index for BG/FS trails
        self.scatters_collected_this_sequence: Set[Tuple[int, int]] = set()

        # FS specific state
        self.fs_base_multiplier: int = 1
        self.fs_ew_collected_session: int = 0
        self.fs_ew_collected_this_spin: int = 0
        self.fs_pending_upgrades: int = 0

    def initialize_spin(self):
        """Resets the state for the start of a new spin sequence."""
        # Reset multiplier for Base Game
        if not self.is_free_spins:
            self.current_multiplier = config.AVALANCHE_MULTIPLIERS_BG[0]
            self.current_multiplier_index = 0
        else:
            # Reset to the start of the current FS trail
            fs_trail = config.FS_MULTIPLIER_TRAILS.get(self.fs_base_multiplier, config.FS_MULTIPLIER_TRAILS[1])
            self.current_multiplier = fs_trail[0]
            self.current_multiplier_index = 0

        self.scatters_collected_this_sequence = set()
        self.fs_ew_collected_this_spin = 0
        # Handle FS upgrades at the start of the *next* spin
        if self.fs_pending_upgrades > 0:
            for _ in range(self.fs_pending_upgrades):
                # Increase base multiplier level
                current_level_idx = config.FS_BASE_MULTIPLIER_LEVELS.index(self.fs_base_multiplier)
                next_level_idx = min(current_level_idx + 1, len(config.FS_BASE_MULTIPLIER_LEVELS) - 1)
                self.fs_base_multiplier = config.FS_BASE_MULTIPLIER_LEVELS[next_level_idx]
                # Add extra spin
                self.remaining_spins += config.FS_SPINS_PER_UPGRADE
            self.fs_pending_upgrades = 0 # Clear pending upgrades


    def update_after_clusters(self, clusters: List, did_explode: bool):
        """Updates the multiplier based on whether winning clusters occurred."""
        if clusters: # Increment only if there were winning clusters
            self.current_multiplier_index += 1
            if self.is_free_spins:
                fs_trail = config.FS_MULTIPLIER_TRAILS.get(self.fs_base_multiplier, config.FS_MULTIPLIER_TRAILS[1])
                safe_index = min(self.current_multiplier_index, len(fs_trail) - 1)
                self.current_multiplier = fs_trail[safe_index]
            else:
                bg_multipliers = config.AVALANCHE_MULTIPLIERS_BG
                safe_index = min(self.current_multiplier_index, len(bg_multipliers) - 1)
                self.current_multiplier = bg_multipliers[safe_index]
        # EW explosions alone do not increase the multiplier index (GDD 4.6)

    def accumulate_scatter(self, coord: Tuple[int, int]):
        """Adds a unique scatter coordinate found during the spin sequence."""
        self.scatters_collected_this_sequence.add(coord)

    def accumulate_ew_collected(self, count: int):
        """Adds EWs collected during a step (relevant for FS)."""
        if self.is_free_spins:
            self.fs_ew_collected_session += count
            self.fs_ew_collected_this_spin += count

    def finalize_spin_sequence(self) -> Tuple[bool, int]:
        """Checks for FS trigger/retrigger at the end of a full sequence."""
        num_scatters = len(self.scatters_collected_this_sequence)
        triggered_fs = False
        spins_won = 0

        if not self.is_free_spins: # Check for Base Game trigger
            if num_scatters >= 3:
                triggered_fs = True
                spins_won = config.FS_TRIGGER_SCATTERS.get(num_scatters)
                if spins_won is None: # Handle 5+
                    max_defined = max(config.FS_TRIGGER_SCATTERS.keys())
                    spins_won = config.FS_TRIGGER_SCATTERS[max_defined] + \
                                (num_scatters - max_defined) * config.FS_TRIGGER_SCATTERS_EXTRA
                # Transition state if triggered
                self.is_free_spins = True
                self.remaining_spins = spins_won
                self.fs_base_multiplier = 1 # Start FS at 1x base
                self.fs_ew_collected_session = 0
                self.fs_pending_upgrades = 0
        else: # Check for Free Spins Retrigger
            if num_scatters >= 2:
                spins_won = config.FS_RETRIGGER_SCATTERS.get(num_scatters)
                if spins_won is None: # Handle 5+
                    max_defined = max(config.FS_RETRIGGER_SCATTERS.keys())
                    spins_won = config.FS_RETRIGGER_SCATTERS[max_defined] + \
                                (num_scatters - max_defined) * config.FS_RETRIGGER_SCATTERS_EXTRA
                self.remaining_spins += spins_won
            # Check for FS upgrades based on EWs collected *this spin*
            # Note: GDD says check happens at end of spin, applies at start of next
            upgrades_earned = self.fs_ew_collected_this_spin // config.FS_EW_COLLECTION_PER_UPGRADE
            self.fs_pending_upgrades += upgrades_earned

        return triggered_fs, spins_won

    def start_free_spins(self, initial_spins: int):
        """Explicitly starts the Free Spins mode."""
        self.is_free_spins = True
        self.remaining_spins = initial_spins
        self.fs_base_multiplier = 1
        self.fs_ew_collected_session = 0
        self.fs_pending_upgrades = 0
        self.initialize_spin() # Set initial FS multiplier etc.

    def consume_free_spin(self):
        """Decrements the remaining free spins count."""
        if self.is_free_spins:
            self.remaining_spins -= 1
            if self.remaining_spins <= 0:
                self.is_free_spins = False # End FS mode 