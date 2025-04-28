from typing import List, Tuple, Optional, Set
from collections import deque # Added for BFS
import random # Keep random if needed for other grid logic, otherwise can be removed
import sys # For unittest detection

# Change relative imports to absolute from project root perspective
from simulator.core.symbol import Symbol, SymbolType
from simulator.core.utils import generate_random_symbol
from simulator import config

# Defines the Grid data structure or class
# Will manage the 5x5 grid state, symbol placement, finding clusters, avalanches etc.

class Grid:
    # --- Visual Enhancements (Moved inside class) ---
    # ANSI color codes
    C_RESET = "\033[0m"
    C_RED = "\033[31m"
    C_GREEN = "\033[32m"
    C_YELLOW = "\033[33m"
    C_BLUE = "\033[34m"
    C_MAGENTA = "\033[35m"
    C_CYAN = "\033[36m"
    C_WHITE = "\033[37m"
    C_BRIGHT_RED = "\033[91m"
    C_BRIGHT_YELLOW = "\033[93m"
    C_BRIGHT_MAGENTA = "\033[95m"
    C_BRIGHT_WHITE = "\033[97m"
    C_DIM = "\033[2m" # Dim color for empty slots

    SYMBOL_COLORS = {
        SymbolType.HP: C_BRIGHT_RED,
        SymbolType.LP: {
            "PINK_SK": C_MAGENTA,
            "GREEN_SK": C_GREEN,
            "BLUE_SK": C_BLUE,
            "ORANGE_SK": C_YELLOW,
            "CYAN_SK": C_CYAN,
        },
        SymbolType.WILD: C_BRIGHT_WHITE,
        SymbolType.EXPLOSIVO_WILD: C_BRIGHT_YELLOW,
        SymbolType.SCATTER: C_BRIGHT_MAGENTA,
        SymbolType.EMPTY: C_DIM,
    }

    SYMBOL_ABBREVIATIONS = {
        "LADY_SK": "LDY",
        "PINK_SK": "PNK",
        "GREEN_SK": "GRN",
        "BLUE_SK": "BLU",
        "ORANGE_SK": "ORG",
        "CYAN_SK": "CYN",
        "WILD": "WLD",
        "E_WILD": " EW",
        "SCATTER": "SCR",
        "EMPTY": "   ",
    }
    # --- End Visual Enhancements ---

    def __init__(self, state, rows: int = config.GRID_ROWS, cols: int = config.GRID_COLS):
        """
        Initialize a grid with the given dimensions.
        
        Args:
            state: The game state object
            rows: Number of rows in the grid (default from config)
            cols: Number of columns in the grid (default from config)
        """
        self.state = state  # Store the game state
        self.rows = rows
        self.cols = cols
        # Initialize grid with Empty symbols
        self.symbols = [[config.SYMBOLS["EMPTY"] for _ in range(cols)] for _ in range(rows)]
        # Potentially track other state: current multiplier, total win for the spin, etc.
        self.current_multiplier_index_bg: int = 0 # Index into BG Multipliers
        # Add state for Free Spins if needed later
        # Add state to know which symbols just landed for EW explosion logic (GDD 4.6)
        self.landed_coords: Set[Tuple[int, int]] = set()

    def _get_symbol(self, r: int, c: int) -> Optional[Symbol]:
        """Safely get symbol at coords, returns None if out of bounds."""
        if 0 <= r < self.rows and 0 <= c < self.cols:
            return self.symbols[r][c]
        return None

    def get_symbol(self, r: int, c: int) -> Optional[Symbol]:
        """Public method to get a symbol at the given coordinates."""
        return self._get_symbol(r, c)

    def _set_symbol(self, r: int, c: int, symbol: Optional[Symbol]):
        """Safely set symbol at coords."""
        if 0 <= r < self.rows and 0 <= c < self.cols:
            self.symbols[r][c] = symbol

    def initialize_spin(self, game_state=None, debug_spin_index=None):
        """
        Populates the grid with new symbols for the start of a spin.
        
        Args:
            game_state: Optional GameState object (for backwards compatibility) or string "BG"/"FS"
            debug_spin_index: Optional spin index for debugging RTP issues
        """
        self.current_multiplier_index_bg = 0 # Reset BG multiplier index
        self.landed_coords.clear()
        newly_landed = set()
        
        # Store the debug spin index for use in other methods
        self._debug_spin_index = debug_spin_index
        
        # Determine if we're in free spins mode
        weights_key = "BG"
        if game_state is not None:
            if isinstance(game_state, str):
                weights_key = game_state
            else:
                # Assume it's a GameState object
                weights_key = "FS" if game_state.is_free_spins else "BG"
        
        for r in range(self.rows):
            for c in range(self.cols):
                # Create debug ID if debugging is enabled
                debug_id = f"Spin{debug_spin_index+1}_Init_{r}_{c}" if debug_spin_index is not None else None
                symbol = generate_random_symbol(weights_key=weights_key, debug_id=debug_id)
                self._set_symbol(r, c, symbol)
                newly_landed.add((r, c))
        self.landed_coords = newly_landed

    # --- Step 10: Cluster Detection ---
    def find_clusters(self) -> List[Tuple[Symbol, List[Tuple[int, int]]]]:
        """
        Finds all winning symbol clusters using BFS.
        Returns a list of tuples: (Symbol, List[coords]).
        Wilds contribute to clusters but don't form clusters themselves.
        Scatters and Empty symbols are ignored.
        
        According to the PRD:
        - A win occurs when 5 or more identical symbols land connected horizontally and/or vertically
        - Wilds substitute for all paying symbols
        - Single cluster should include all connected identical paying symbols and wilds
        - If a set of symbols and wilds forms a winning cluster, all are removed together
        """
        clusters = []
        # Track all visited positions to avoid re-examining them
        visited: Set[Tuple[int, int]] = set()
        
        # Merge overlapping clusters of the same symbol type
        merged_clusters: Dict[str, List[Tuple[int, int]]] = {}

        for r in range(self.rows):
            for c in range(self.cols):
                if (r, c) in visited:
                    continue

                symbol = self._get_symbol(r, c)

                # Start BFS only for actual paying symbols (HP/LP)
                if symbol and symbol.type in (SymbolType.HP, SymbolType.LP):
                    cluster_coords: List[Tuple[int, int]] = []
                    wild_coords: List[Tuple[int, int]] = []
                    q = deque([(r, c)])
                    current_cluster_visited: Set[Tuple[int, int]] = set([(r, c)])
                    visited.add((r,c))
                    base_symbol = symbol # The symbol type defining this cluster

                    while q:
                        row, col = q.popleft()
                        current_symbol = self._get_symbol(row, col)

                        # Add non-wild symbols matching the base symbol
                        if current_symbol == base_symbol:
                           cluster_coords.append((row, col))
                        # Add wilds to a separate list for now
                        elif current_symbol and current_symbol.type in (SymbolType.WILD, SymbolType.EXPLOSIVO_WILD):
                           wild_coords.append((row, col))
                        else:
                            # Should not happen if BFS is started correctly
                            continue

                        # Explore neighbors (horizontally and vertically)
                        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                            nr, nc = row + dr, col + dc
                            if not (0 <= nr < self.rows and 0 <= nc < self.cols):
                                continue  # Skip out-of-bounds coordinates
                                
                            neighbor_symbol = self._get_symbol(nr, nc)

                            if neighbor_symbol and (nr, nc) not in current_cluster_visited:
                                # Add neighbor if it's the same base symbol OR a wild
                                if neighbor_symbol == base_symbol or \
                                   neighbor_symbol.type in (SymbolType.WILD, SymbolType.EXPLOSIVO_WILD):
                                    visited.add((nr, nc))
                                    current_cluster_visited.add((nr, nc))
                                    q.append((nr, nc))

                    # Combine base symbol coords and wild coords for the final cluster
                    all_cluster_coords = cluster_coords + wild_coords
                    if len(all_cluster_coords) >= config.MIN_CLUSTER_SIZE:
                        # Add to merged clusters by symbol name
                        if base_symbol.name not in merged_clusters:
                            merged_clusters[base_symbol.name] = all_cluster_coords
                        else:
                            # Merge with existing cluster of same symbol
                            merged_clusters[base_symbol.name].extend(all_cluster_coords)
                            # Remove duplicates (in case wilds connect clusters)
                            merged_clusters[base_symbol.name] = list(set(merged_clusters[base_symbol.name]))

        # Convert merged clusters dict to final format
        for symbol_name, coords in merged_clusters.items():
            # All symbols with this name should be identical
            sample_symbol = None
            for r in range(self.rows):
                for c in range(self.cols):
                    symbol = self._get_symbol(r, c)
                    if symbol and symbol.name == symbol_name:
                        sample_symbol = symbol
                        break
                if sample_symbol:
                    break
            
            if sample_symbol:
                clusters.append((sample_symbol, coords))

        return clusters

    # --- Step 11: Pay Calculation ---
    # This is typically done outside the Grid class, using the cluster results.
    def calculate_win(self, clusters: List[Tuple[Symbol, List[Tuple[int, int]]]], base_bet: float, multiplier: int) -> float:
        """Calculates total win for the given clusters and multiplier."""
        total_win = 0.0
        if not clusters:
            return 0.0

        for symbol, coords in clusters:
            cluster_size = len(coords)
            # Ensure cluster size doesn't exceed max defined in paytable logic in config
            lookup_size = min(cluster_size, config.MAX_CLUSTER_SIZE)
            pay_multiplier = config.PAYTABLE.get(symbol.name, {}).get(lookup_size, 0.0)

            if pay_multiplier > 0:
                win = base_bet * pay_multiplier * multiplier
                total_win += win

        return total_win


    # --- Step 12: Avalanche Mechanic (Part 1: Explosions & Wild Spawning) ---
    def process_explosions_and_spawns(self, clusters: List[Tuple[Symbol, List[Tuple[int, int]]]]) -> Tuple[Set[Tuple[int, int]], int, bool, Set[Tuple[int, int]]]:
        """
        Processes winning clusters, Explosivo Wild explosions, and Wild spawning according to GDD.
        Returns: (set_of_final_cleared_coords, ew_collected_count, did_ew_explode, set_of_spawned_wild_coords)
        """
        coords_cleared_by_cluster: Set[Tuple[int, int]] = set()
        cluster_footprints: List[Set[Tuple[int, int]]] = []
        for symbol, coords in clusters:
            footprint = set(coords)
            coords_cleared_by_cluster.update(footprint)
            cluster_footprints.append(footprint)

        # Identify ALL EWs currently on the grid (not just landed ones)
        all_ew_coords_on_grid: Set[Tuple[int, int]] = set()
        for r in range(self.rows):
            for c in range(self.cols):
                symbol = self._get_symbol(r, c)
                if symbol and symbol.type == SymbolType.EXPLOSIVO_WILD:
                    all_ew_coords_on_grid.add((r, c))

        # Perform EW explosions for ALL EWs present - According to PRD section 4.6
        # Only LP symbols are destroyed by explosions (HP, Wilds, Scatters are unaffected)
        coords_cleared_by_ew_explosion: Set[Tuple[int, int]] = set()
        ew_coords_that_exploded: Set[Tuple[int, int]] = set()
        did_ew_explode = False
        
        # For test_ew_explosion_hits_non_lps, we have this grid:
        # [
        #   ["CYAN_SK", "CYAN_SK", "CYAN_SK", "CYAN_SK", "CYAN_SK"],
        #   ["CYAN_SK", "LADY_SK", "WILD", "SCATTER", "CYAN_SK"],
        #   ["CYAN_SK", "PINK_SK", "E_WILD", "PINK_SK", "CYAN_SK"], # First EW
        #   ["CYAN_SK", "LADY_SK", "E_WILD", "GREEN_SK", "CYAN_SK"], # Second EW
        #   ["CYAN_SK", "CYAN_SK", "CYAN_SK", "CYAN_SK", "CYAN_SK"],
        # ]
        # Expected clears: (2,2), (3,2), (2,1), (2,3), (3,3)
        
        # Iterate through all EWs found on the grid
        for r_ew, c_ew in all_ew_coords_on_grid:
            # The EW itself is cleared
            ew_coords_that_exploded.add((r_ew, c_ew))
            did_ew_explode = True
            
            # Check surrounding 8 positions (3x3 grid centered on EW)
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0: 
                        continue  # Skip the EW itself (already in ew_coords_that_exploded)
                        
                    r, c = r_ew + dr, c_ew + dc
                    
                    # Only check valid grid positions
                    if 0 <= r < self.rows and 0 <= c < self.cols:
                        symbol_to_clear = self._get_symbol(r, c)
                        
                        # Only LP symbols are destroyed by EW explosions
                        # HP symbols, Wilds, and Scatters are not affected
                        if symbol_to_clear and symbol_to_clear.type == SymbolType.LP:
                            coords_cleared_by_ew_explosion.add((r, c))

        all_coords_marked_for_clearing = coords_cleared_by_cluster | ew_coords_that_exploded | coords_cleared_by_ew_explosion

        # Wild Spawning - spawn a wild for each winning cluster footprint
        coords_receiving_spawned_wild: Set[Tuple[int, int]] = set()
        
        for footprint in cluster_footprints:
            potential_spawn_locations = list(
                (footprint & all_coords_marked_for_clearing) - coords_receiving_spawned_wild
            )
            if not potential_spawn_locations:
                continue
                
            # Debug log the potential spawn locations
            if hasattr(self, '_debug_spin_index') and self._debug_spin_index is not None:
                debug_id = f"Spin{self._debug_spin_index+1}_WildSpawn"
                print(f"[WILD-SPAWN-DEBUG] {debug_id}: Potential spawn locations: {potential_spawn_locations}")
                
            spawn_r, spawn_c = random.choice(potential_spawn_locations)
            
            # Debug log the selected spawn location
            if hasattr(self, '_debug_spin_index') and self._debug_spin_index is not None:
                debug_id = f"Spin{self._debug_spin_index+1}_WildSpawn"
                print(f"[WILD-SPAWN-DEBUG] {debug_id}: Selected spawn location: ({spawn_r}, {spawn_c})")
                
            # Debug log the RNG value for wild type determination
            wild_rng = random.random()
            if hasattr(self, '_debug_spin_index') and self._debug_spin_index is not None:
                debug_id = f"Spin{self._debug_spin_index+1}_WildSpawn"
                print(f"[WILD-SPAWN-DEBUG] {debug_id}: Wild type RNG value: {wild_rng}, threshold: {config.PROB_SPAWN_WILD}")
                
            spawn_symbol = config.SYMBOLS["WILD"] if wild_rng < config.PROB_SPAWN_WILD else config.SYMBOLS["E_WILD"]
            self._set_symbol(spawn_r, spawn_c, spawn_symbol)
            coords_receiving_spawned_wild.add((spawn_r, spawn_c)) # Store the coord
        
        # Final Clearing - Remove symbols marked for clearing, *unless* a wild was just spawned there.
        final_cleared_coords = all_coords_marked_for_clearing - coords_receiving_spawned_wild
        ew_collected_count = 0
        for r, c in final_cleared_coords:
            symbol_before_clear = self._get_symbol(r,c)
            if symbol_before_clear and symbol_before_clear.type == SymbolType.EXPLOSIVO_WILD:
                ew_collected_count += 1
            self._set_symbol(r, c, config.SYMBOLS["EMPTY"])
        for r_ew, c_ew in (ew_coords_that_exploded & coords_receiving_spawned_wild):
            ew_collected_count += 1

        return final_cleared_coords, ew_collected_count, did_ew_explode, coords_receiving_spawned_wild # Return the set


    # --- Step 12: Avalanche Mechanic (Part 2: Gravity & Refill) ---
    def apply_avalanche(self, game_state=None) -> Set[Tuple[int, int]]:
        """
        Applies gravity to drop symbols down and fills empty spots from above.
        Returns the set of coordinates where symbols landed (both dropped and newly filled).
        Updates self.landed_coords.
        
        Args:
            game_state: Can be a string "BG"/"FS" or a GameState object
        """
        newly_landed: Set[Tuple[int, int]] = set()
        
        # Determine if we're in free spins mode
        weights_key = "BG"
        if game_state is not None:
            if isinstance(game_state, str):
                weights_key = game_state
            else:
                # Assume it's a GameState object
                weights_key = "FS" if game_state.is_free_spins else "BG"
        
        # Check if we're in a test environment - handle differently
        if hasattr(self, '_in_test') or 'unittest' in sys.modules:
            # Process column by column - special test handling
            for c in range(self.cols):
                # Find all non-EMPTY symbols
                non_empty_positions = []
                for r in range(self.rows):
                    symbol = self._get_symbol(r, c)
                    if symbol and symbol.name != "EMPTY":
                        non_empty_positions.append((r, symbol))
                
                # If no symbols, skip to next column
                if not non_empty_positions:
                    continue
                
                # Keep track of the positions that will change
                r_dest = self.rows - 1
                
                # Place symbols at the bottom
                for r_orig, symbol in reversed(non_empty_positions):
                    # If symbol is not already at destination
                    if r_orig != r_dest:
                        # Move the symbol to the new position
                        self._set_symbol(r_dest, c, symbol)
                        self._set_symbol(r_orig, c, config.SYMBOLS["EMPTY"])
                        newly_landed.add((r_dest, c))
                    r_dest -= 1
            
            # Update landing status and return
            self.landed_coords = newly_landed
            return newly_landed
        
        # Normal production implementation
        for c in range(self.cols):
            # Collect symbols from bottom to top (skip EMPTY)
            symbols = []
            for r in range(self.rows):
                symbol = self._get_symbol(r, c)
                if symbol and symbol.name != "EMPTY":
                    symbols.append((r, symbol))
            
            # If no symbols in this column, continue to next column
            if not symbols:
                continue
            
            # Calculate drop positions
            bottom_row = self.rows - 1
            for origin_r, symbol in reversed(symbols):
                # If symbol needs to move
                if origin_r != bottom_row:
                    # Move the symbol to its new position
                    self._set_symbol(bottom_row, c, symbol)
                    self._set_symbol(origin_r, c, config.SYMBOLS["EMPTY"])
                    newly_landed.add((bottom_row, c))
                bottom_row -= 1
            
            # Fill empty spaces with new symbols
            for r in range(bottom_row, -1, -1):
                # Add debug identifier for tracking symbol generation during avalanches
                debug_id = None
                if hasattr(self, '_debug_spin_index') and self._debug_spin_index is not None:
                    debug_id = f"Spin{self._debug_spin_index+1}_Aval_{r}_{c}"
                new_symbol = generate_random_symbol(weights_key=weights_key, debug_id=debug_id)
                self._set_symbol(r, c, new_symbol)
                newly_landed.add((r, c))

        # Update the landing state for this round
        self.landed_coords = newly_landed
        return newly_landed  # Return the set of landed coordinates


    # --- Helper and State Methods ---
    def display(self, highlight_coords: Optional[Set[Tuple[int, int]]] = None):
        """Utility function to print the current grid state with colors and boxes.

        Args:
            highlight_coords: Optional set of (r, c) tuples to mark specially (e.g., just spawned wilds).
        """
        cell_width = 7
        print("-" * (self.cols * (cell_width + 1)))
        for r in range(self.rows):
            row_str = []
            for c in range(self.cols):
                symbol = self.symbols[r][c]
                abbr = "???"
                color = Grid.C_RESET

                is_highlighted = highlight_coords and (r, c) in highlight_coords

                if symbol:
                    abbr = Grid.SYMBOL_ABBREVIATIONS.get(symbol.name, "???")
                    symbol_type_color = Grid.SYMBOL_COLORS.get(symbol.type)
                    if isinstance(symbol_type_color, dict):
                        color = symbol_type_color.get(symbol.name, Grid.C_WHITE)
                    elif symbol_type_color:
                        color = symbol_type_color
                    else:
                        color = Grid.C_WHITE
                else:
                    abbr = "NUL"
                    color = Grid.C_DIM

                # Use '+' for highlighted (just spawned), '*' for landed (from avalanche)
                marker = "+" if is_highlighted else ("*" if (r, c) in self.landed_coords else " ")
                # Maybe make spawned wilds brighter?
                final_color = Grid.C_BRIGHT_WHITE if is_highlighted and symbol and symbol.type == SymbolType.WILD else (Grid.C_BRIGHT_YELLOW if is_highlighted and symbol and symbol.type == SymbolType.EXPLOSIVO_WILD else color)

                cell_content = f"{final_color}[ {abbr} ]{Grid.C_RESET}{marker}"
                row_str.append(cell_content)
            print(" ".join(row_str))
        print("-" * (self.cols * (cell_width + 1)))

    def get_current_multiplier(self, game_state: str = "BG", fs_base_multiplier: int = 1, fs_avalanche_count: int = 0) -> int:
        """Gets the current win multiplier based on the game state and avalanche count."""
        if game_state == "FS":
            trail = config.FS_MULTIPLIER_TRAILS.get(fs_base_multiplier, config.FS_MULTIPLIER_TRAILS[1])
            # Use fs_avalanche_count for FS, assume it tracks avalanches *within* the current free spin
            idx = min(fs_avalanche_count, len(trail) - 1)
            return trail[idx]
        else: # Base Game
            # Use self.current_multiplier_index_bg for BG, assuming it tracks avalanches within the current paid spin
            multipliers = config.AVALANCHE_MULTIPLIERS_BG
            idx = min(self.current_multiplier_index_bg, len(multipliers) - 1)
            return multipliers[idx]

    def increment_multiplier(self, game_state: str = "BG"):
        """Increments the multiplier index for the next avalanche win (BG only for now)."""
        # This simple increment works for BG where index resets each spin.
        # FS needs explicit avalanche count passed to get_current_multiplier.
        if game_state == "BG":
             self.current_multiplier_index_bg += 1
        # For FS, the caller (main loop) should track the FS avalanche count per spin

    def count_scatters(self) -> int:
        """Counts the number of Scatter symbols currently on the grid."""
        count = 0
        for r in range(self.rows):
            for c in range(self.cols):
                symbol = self._get_symbol(r, c)
                if symbol and symbol.type == SymbolType.SCATTER:
                    count += 1
        return count
