# import pytest # F401 unused

from simulator.main import run_simulation
from simulator import config # Import config to control JIT for the test

# Golden values for 10,000 spins with seed 20250101 and JIT explicitly disabled
GOLDEN_RTP = 46.0570       # %
GOLDEN_HIT_FREQ = 17.85    # %
# GOLDEN_FS_TRIGGERS = 0 # Current config has SCATTER weight 0 in BG

# Tolerance for floating point comparisons (absolute difference)
RTP_TOLERANCE = 0.05       # ±0.05%
HIT_FREQ_TOLERANCE = 0.05  # ±0.05%

def test_regression_against_golden_values():
    """Regression guardrail: Compares current run against golden values for fixed seed and spin count."""
    num_spins = 10000
    seed = 20250101
    
    # Ensure JIT is disabled for this regression test to match golden value generation
    original_jit_state = config.ENABLE_JIT
    config.ENABLE_JIT = False
    
    try:
        current_stats = run_simulation(
            num_spins=num_spins,
            base_bet=1.0,
            run_id="reg_test_run",
            return_stats=True,
            verbose_spins=0, # No verbose logs for regression test speed
            seed=seed,
            calc_roe_flag=False,      # ROE not part of this core regression value
            enhanced_stats=False    # Basic stats are sufficient
        )
    finally:
        config.ENABLE_JIT = original_jit_state # Restore JIT state

    assert current_stats is not None, "run_simulation did not return stats."
    if "error" in current_stats:
        raise AssertionError(f"Simulation failed with error: {current_stats['error']} after {current_stats.get('total_spins_completed', 0)} spins")

    # Compare RTP
    rtp_current = current_stats.get('rtp')
    assert rtp_current is not None, "RTP not found in current simulation stats."
    assert abs(rtp_current - GOLDEN_RTP) <= RTP_TOLERANCE, \
        f"RTP drift too high: Current={rtp_current:.4f}%, Golden={GOLDEN_RTP:.4f}%"

    # Compare hit frequency
    hit_freq_current = current_stats.get('hit_frequency')
    assert hit_freq_current is not None, "Hit Frequency not found in current simulation stats."
    assert abs(hit_freq_current - GOLDEN_HIT_FREQ) <= HIT_FREQ_TOLERANCE, \
        f"Hit Frequency drift too high: Current={hit_freq_current:.2f}%, Golden={GOLDEN_HIT_FREQ:.2f}%"

    # Placeholder for future: Add FS triggers if scatter weights become non-zero
    # fs_triggers_current = current_stats.get('fs_triggers')
    # assert fs_triggers_current is not None, "FS Triggers not found in current simulation stats."
    # assert fs_triggers_current == 0, \
    #     f"FS Triggers drift: Current={fs_triggers_current}, Golden=0 (expected with current weights)"

# Remove or comment out the old consistency test if this replaces it
# def test_regression_rtp_and_hit_rate_consistency():
#     ... (old code)
