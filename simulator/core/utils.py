import random
from typing import Dict, List, Tuple
from functools import lru_cache

# Change relative imports to absolute from project root perspective
from simulator import config
from simulator.core.symbol import Symbol

@lru_cache(maxsize=2) # Cache results for BG and FS weights
def _prepare_weights(weights_dict: Tuple[Tuple[str, int], ...]) -> Tuple[List[str], List[int]]:
    """Helper to convert weight dict (as tuple for caching) to lists."""
    weights_dict_actual = dict(weights_dict)
    symbol_names = list(weights_dict_actual.keys())
    symbol_weights = list(weights_dict_actual.values())
    total_weight = sum(symbol_weights)
    if total_weight <= 0:
        raise ValueError(f"Total weight of symbols must be positive in weights dict: {weights_dict_actual}")
    return symbol_names, symbol_weights

def generate_random_symbol(weights_key: str = "BG", sim_type: str = "main", debug_id: str = None) -> Symbol:
    """
    Generates a random symbol based on the specified weights (BG or FS) from config.
    
    For test mocking, this function can return pre-specified symbols from mock_choices.
    
    Args:
        weights_key: "BG" or "FS" to select which weight set to use
        sim_type: "main" or "optimized" to use consistent RNG method
        debug_id: Optional identifier for debugging RTP issues
        
    Returns:
        A Symbol object from config.SYMBOLS
    """
    # Get appropriate weights for BG or FS
    if weights_key == "FS":
        weights_dict = config.SYMBOL_GENERATION_WEIGHTS_FS
    else:  # Default to Base Game
        weights_dict = config.SYMBOL_GENERATION_WEIGHTS_BG
    
    # Convert dict to tuple of items for caching
    weights_tuple = tuple(sorted(weights_dict.items()))
    symbol_names, symbol_weights = _prepare_weights(weights_tuple)
    
    # Generate a random symbol using a deterministic approach
    chosen_name = random.choices(symbol_names, weights=symbol_weights, k=1)[0]
    
    # Debug logging for RTP investigation
    if debug_id is not None:
        print(f"[SYMBOL-DEBUG] {debug_id}: Generated symbol {chosen_name} (using {weights_key} weights)")
        # Log the current RNG state
        state_hash = hash(str(random.getstate())[:100])  # Just use part of the state to keep log manageable
        print(f"[SYMBOL-DEBUG] {debug_id}: RNG state hash: {state_hash}")
    
    return config.SYMBOLS[chosen_name]

# Example usage needs update if run directly
if __name__ == '__main__':
    print("--- Base Game Weights ---")
    counts_bg = {name: 0 for name in config.SYMBOL_GENERATION_WEIGHTS_BG.keys()}
    num_samples = 10000
    for _ in range(num_samples):
        symbol = generate_random_symbol("BG")
        counts_bg[symbol.name] += 1

    total_weight_bg = sum(config.SYMBOL_GENERATION_WEIGHTS_BG.values())
    print(f"BG Symbol generation counts after {num_samples} samples:")
    for name, count in counts_bg.items():
        expected = (config.SYMBOL_GENERATION_WEIGHTS_BG.get(name, 0) / total_weight_bg) * num_samples
        print(f"  {name}: {count} (Expected: ~{expected:.1f})")

    print("\n--- Free Spins Weights ---")
    counts_fs = {name: 0 for name in config.SYMBOL_GENERATION_WEIGHTS_FS.keys()}
    for _ in range(num_samples):
        symbol = generate_random_symbol("FS")
        counts_fs[symbol.name] += 1

    total_weight_fs = sum(config.SYMBOL_GENERATION_WEIGHTS_FS.values())
    print(f"FS Symbol generation counts after {num_samples} samples:")
    for name, count in counts_fs.items():
        expected = (config.SYMBOL_GENERATION_WEIGHTS_FS.get(name, 0) / total_weight_fs) * num_samples
        print(f"  {name}: {count} (Expected: ~{expected:.1f})")
