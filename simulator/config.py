import datetime
import json
import pathlib

import numpy as np  # NEW: vector-friendly weights

from simulator.core.symbol import Symbol, SymbolType

# Configuration file for the simulator
# Based on docs/GDD Math.md

# --- Symbol Definitions ---
# Renamed based on GDD
SYMBOLS = {
    # High Pay (HP)
    "LADY_SK": Symbol(name="LADY_SK", type=SymbolType.HP),  # Symbol 1
    # Low Pay (LP)
    "PINK_SK": Symbol(name="PINK_SK", type=SymbolType.LP),  # Symbol 2
    "GREEN_SK": Symbol(name="GREEN_SK", type=SymbolType.LP),  # Symbol 3
    "BLUE_SK": Symbol(name="BLUE_SK", type=SymbolType.LP),  # Symbol 4
    "ORANGE_SK": Symbol(name="ORANGE_SK", type=SymbolType.LP),  # Symbol 5
    "CYAN_SK": Symbol(name="CYAN_SK", type=SymbolType.LP),  # Symbol 6
    # Wilds & Scatters
    "WILD": Symbol(name="WILD", type=SymbolType.WILD),  # Standard Wild
    "E_WILD": Symbol(name="E_WILD", type=SymbolType.EXPLOSIVO_WILD),  # Explosivo Wild
    "SCATTER": Symbol(name="SCATTER", type=SymbolType.SCATTER),  # Silver Robot Skull
    # Internal representation
    "EMPTY": Symbol(name="EMPTY", type=SymbolType.EMPTY),
}

# --- Paytable Configuration ---
# Payouts are multipliers of the total bet for cluster sizes.
# Based on GDD table, expanding ranges.
PAYTABLE = {
    # Symbol Name: { cluster_size: multiplier }
    "LADY_SK": {
        5: 1.0,
        6: 1.5,
        7: 2.5,
        8: 5.0,
        9: 5.0,
        10: 7.5,
        11: 7.5,
        12: 25.0,
        13: 25.0,
        14: 25.0,
        15: 150.0,
    },
    "PINK_SK": {
        5: 0.5,
        6: 0.7,
        7: 1.0,
        8: 1.7,
        9: 1.7,
        10: 2.5,
        11: 2.5,
        12: 7.5,
        13: 7.5,
        14: 7.5,
        15: 50.0,
    },
    "GREEN_SK": {
        5: 0.4,
        6: 0.7,
        7: 0.8,
        8: 1.4,
        9: 1.4,
        10: 2.0,
        11: 2.0,
        12: 6.0,
        13: 6.0,
        14: 6.0,
        15: 40.0,
    },
    "BLUE_SK": {
        5: 0.3,
        6: 0.5,
        7: 0.6,
        8: 1.0,
        9: 1.0,
        10: 1.5,
        11: 1.5,
        12: 5.0,
        13: 5.0,
        14: 5.0,
        15: 30.0,
    },
    "ORANGE_SK": {
        5: 0.3,
        6: 0.4,
        7: 0.5,
        8: 0.8,
        9: 0.8,
        10: 1.2,
        11: 1.2,
        12: 4.0,
        13: 4.0,
        14: 4.0,
        15: 25.0,
    },
    "CYAN_SK": {
        5: 0.2,
        6: 0.3,
        7: 0.4,
        8: 0.6,
        9: 0.6,
        10: 1.0,
        11: 1.0,
        12: 3.0,
        13: 3.0,
        14: 3.0,
        15: 20.0,
    },
    # Wilds substitute but don't form their own clusters typically.
    # Scatters trigger features, no direct cluster payout.
}
# Add max cluster size check helper
MAX_CLUSTER_SIZE = 15  # Explicitly based on paytable
for symbol_pays in PAYTABLE.values():
    for size in range(MAX_CLUSTER_SIZE + 1, 26):
        symbol_pays[size] = symbol_pays[MAX_CLUSTER_SIZE]

# --- Symbol Distribution / Reel Generation Weights ---
# Defines the probability or weight of each symbol appearing.
# Placeholder values - Requires tuning based on simulation target RTP.
# Separate distributions needed for BG and FS (GDD 4.4)
SYMBOL_GENERATION_WEIGHTS_BG = {
    "LADY_SK": 19.0,  # Updated from 6.5
    "PINK_SK": 26.0,  # Updated from 25.8
    "GREEN_SK": 26.0,  # Updated from 25.8
    "BLUE_SK": 26.0,  # Updated from 25.8
    "ORANGE_SK": 26.0,  # Updated from 25.8
    "CYAN_SK": 26.0,  # Updated from 25.8
    "WILD": 0.0,  # Updated from 9.2
    "E_WILD": 0.0,  # Unchanged
    "SCATTER": 0.0,  # Unchanged
}
SYMBOL_GENERATION_WEIGHTS_FS = {
    "LADY_SK": 21.0,  # Unchanged
    "PINK_SK": 23.0,  # Updated from 22.8
    "GREEN_SK": 23.0,  # Updated from 22.8
    "BLUE_SK": 23.0,  # Updated from 22.8
    "ORANGE_SK": 23.0,  # Updated from 22.8
    "CYAN_SK": 23.0,  # Updated from 22.8
    "WILD": 4.1,  # Updated from 3.0
    "E_WILD": 4.1,  # Unchanged
    "SCATTER": 0.30,  # Updated from 0.40
}

# --- Numpy representations (for fast vector maths) ---
BG_SYMBOL_NAMES: np.ndarray = np.array(list(SYMBOL_GENERATION_WEIGHTS_BG.keys()))
BG_WEIGHTS: np.ndarray = np.array(
    list(SYMBOL_GENERATION_WEIGHTS_BG.values()), dtype=np.float64
)
FS_SYMBOL_NAMES: np.ndarray = np.array(list(SYMBOL_GENERATION_WEIGHTS_FS.keys()))
FS_WEIGHTS: np.ndarray = np.array(
    list(SYMBOL_GENERATION_WEIGHTS_FS.values()), dtype=np.float64
)

# --- Validation helper ---


def validate_config() -> None:
    """Validate that configuration constants are self-consistent.

    Raises
    ------
    ValueError
        If the sum of any weight vector is zero / negative or the symbol lists
        between BG and FS differ.
    """
    if BG_WEIGHTS.sum() <= 0:
        raise ValueError("Total BG symbol weight must be positive")
    if FS_WEIGHTS.sum() <= 0:
        raise ValueError("Total FS symbol weight must be positive")

    if set(BG_SYMBOL_NAMES) != set(FS_SYMBOL_NAMES):
        raise ValueError(
            "BG and FS symbol sets must be identical for vector maths assumptions to hold"
        )


# Perform validation at import time so that misconfigurations fail fast
validate_config()

# --- JIT support ---
try:
    from numba import jit

    JIT_AVAILABLE: bool = True
except ImportError:
    JIT_AVAILABLE: bool = False

    def jit(**kwargs):  # type: ignore
        """Fallback decorator when Numba is not installed."""

        def decorator(func):
            return func

        return decorator


# Flag to enable JIT cluster detection (can be overridden via CLI)
ENABLE_JIT: bool = False

# --- Wild Spawning Probabilities ---
# GDD 4.7 - P(SpawnW) + P(SpawnEW) = 1
# Requires tuning.
PROB_SPAWN_WILD = 0.60
PROB_SPAWN_E_WILD = 0.40

# --- Simulation Parameters ---
GRID_ROWS = 5
GRID_COLS = 5
MIN_CLUSTER_SIZE = 5  # Minimum symbols for a winning cluster (GDD 4.1)
TOTAL_SIMULATION_SPINS = 1_000_000  # Example number of spins to run
BASE_BET = 1.0  # Assume a base bet of 1 unit for payout calculations

# --- Feature Multipliers ---
# Base Game Multipliers (GDD 4.5)
AVALANCHE_MULTIPLIERS_BG = [1, 2, 4, 8, 16, 32]  # 6 steps
# Free Spins Multipliers (GDD 4.5) - Need Base FS multiplier state
# Example trails based on FS Base Multiplier
FS_MULTIPLIER_TRAILS = {
    1: [1, 2, 4, 8, 16, 32],
    2: [2, 4, 8, 16, 32, 64],
    4: [4, 8, 16, 32, 64, 128],
    8: [8, 16, 32, 64, 128, 256],
    16: [16, 32, 64, 128, 256, 512],
    32: [32, 64, 128, 256, 512, 1024],
}
# Alias for backward compatibility with tests
FS_MUCHO_MULTIPLIER_TRAIL = FS_MULTIPLIER_TRAILS
FS_BASE_MULTIPLIER_LEVELS = list(FS_MULTIPLIER_TRAILS.keys())  # [1, 2, 4, 8, 16, 32]
MAX_FS_BASE_MULTIPLIER = 32

# --- Free Spins Parameters (GDD 4.8, 4.9) ---
FS_TRIGGER_SCATTERS = {3: 10, 4: 12}  # Scatters: Spins
FS_TRIGGER_SCATTERS_EXTRA = 2  # Spins per scatter above 4
FS_RETRIGGER_SCATTERS = {2: 3, 3: 5, 4: 7}  # Scatters: Extra Spins
FS_RETRIGGER_SCATTERS_EXTRA = 2  # Extra spins per scatter above 4
FS_EW_COLLECTION_PER_UPGRADE = 3  # EWs needed per Base Multiplier upgrade
FS_SPINS_PER_UPGRADE = 1  # Extra spins awarded per upgrade

# --- Bet+ Options (GDD 4.10) ---
# Costs as multipliers of Base Bet
BETPLUS_BONUS_BOOST_COST = 1.5
BETPLUS_ENRICO_SHOW_COST = 2.0
BETPLUS_BONUS_BOOST_PLUS_COST = 3.0
# Effects need to be implemented via modifying weights/logic

# --- Feature Buy (GDD 4.11) ---
FEATURE_BUY_COST = 75.0  # Multiplier of Base Bet
FEATURE_BUY_SPINS = 10

MAX_WIN_CAP_MULTIPLIER = 7500.0  # Overall max win cap as a multiplier of the base bet

# --- ROE Simulation Parameters ---
ROE_NUM_SIMULATIONS = 1000  # Number of simulations to run for ROE calculation
ROE_MAX_SPINS = (
    1_000_000  # Max spins per ROE simulation before it's considered infinite
)

# Assign compact int8 IDs to every symbol (Epic 5 – performance refactor)
SYMBOL_TO_ID: dict[str, int] = {}
ID_TO_SYMBOL: list[Symbol] = []
for _idx, _name in enumerate(SYMBOLS.keys()):
    SYMBOL_TO_ID[_name] = _idx
    ID_TO_SYMBOL.append(SYMBOLS[_name])
    # inject the id onto the immutable Symbol via object.__setattr__
    object.__setattr__(SYMBOLS[_name], "id", _idx)  # type: ignore[attr-defined]

SYMBOL_ID_DTYPE = np.int8  # public constant – used by Grid numeric buffers


def save_weights(path: str | pathlib.Path) -> None:
    """Save the current BG/FS weight blobs to *path* in JSON format.

    The file contains both BG and FS dictionaries plus a minimal metadata
    section (timestamp & simulator version placeholder).
    """
    payload = {
        "generated_at": datetime.datetime.utcnow().isoformat() + "Z",
        "bg": SYMBOL_GENERATION_WEIGHTS_BG,
        "fs": SYMBOL_GENERATION_WEIGHTS_FS,
    }
    path = pathlib.Path(path)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
