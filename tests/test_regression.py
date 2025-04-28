import random
import pytest
from simulator.main import run_simulation
from simulator.optimized import run_optimized_simulation

@ pytest.mark.parametrize("num_spins, tolerance", [
    (100, 1e-6),  # small run, exact match
    (500, 1e-6),  # moderate run, exact match
])
def test_regression_main_vs_optimized(num_spins, tolerance):
    """Regression test: main and optimized implementations should produce identical stats with same seed."""
    seed = 42
    random.seed(seed)
    # Run sequential version
    main_stats = run_simulation(
        num_spins=num_spins,
        base_bet=1.0,
        run_id="reg_run",
        verbose_spins=0,
        verbose_fs_only=False,
        return_stats=True,
        enhanced_stats=False
    )
    # Run optimized version with same seed and single-core, no GPU, no JIT
    random.seed(seed)
    optimized_stats = run_optimized_simulation(
        num_spins=num_spins,
        base_bet=1.0,
        run_id="reg_run_opt",
        verbose_spins=0,
        verbose_fs_only=False,
        return_stats=True,
        enhanced_stats=False,
        batch_size=num_spins,
        use_gpu=False,
        create_plots=False,
        cores=1,
        enable_jit=False,
        turbo_mode=False,
        calculate_roe=False,
        roe_use_main_data=False,
        roe_num_sims=0
    )
    # Compare key metrics
    for key in ['total_spins', 'total_win', 'rtp', 'hit_count', 'hit_frequency']:
        assert pytest.approx(main_stats[key], abs=tolerance) == optimized_stats[key], \
            f"Mismatch in {key}: {main_stats[key]} vs {optimized_stats[key]}" 