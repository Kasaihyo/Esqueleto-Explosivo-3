# TODO: Investigate Simulation Discrepancies (Optimized vs. Unoptimized)

This list outlines steps to diagnose why the unoptimized simulation (`simulator.main`) ran faster than the optimized one (`simulator.optimized`) and why their reported RTP values are significantly different based on the runs on 2025-04-20.

- [x] **1. Standardize Run Configuration:**
    - [x] Run both simulations with the *exact same* parameters for a fair comparison:
        - [x] Use a fixed seed (`--seed 42`) for reproducibility.
        - [x] Disable ROE calculation (`--no-roe` for optimized, confirmed standard runs separate ROE sims by default).
        - [x] Use a smaller number of spins (100,000) initially for faster testing.
        - [x] Use `--stats-only` consistently.
    - [x] Command examples executed (e.g., `optimized_test_1`, `standard_test_1`, etc.).

- [x] **2. Compare Configurations:**
    - [x] Review `simulator/config.py` for default values (Found 0 weights for Wilds/Scatters).
    - [x] Check how `simulator/main.py` and `simulator/optimized.py` load/override these configurations (Both import `config`, `main.py` doesn't seem to modify weights).
    - [x] Check `grid.py` and `utils.py` - confirmed they use config correctly.

- [x] **3. Analyze Performance:**
    - [x] Speed difference persists after standardization (`main.py` core ~6s, `optimized.py` ~24s for 100k spins).
    - [ ] Profile both scripts using `cProfile` to identify bottlenecks.
    - [x] Review the optimization logic in `simulator/optimized.py` (Imports core logic from `main.py`, uses `joblib` parallelization, Numba functions exist but usage unclear in main loop).
    - [x] Check if the optimized version is actually *using* the optimizations (Logs indicate hardware detection active, but performance is worse).
    - [x] Test optimized version with `--cores 1` - performance still slow, suggesting overhead isn't solely parallelism.

- [x] **4. Analyze RTP Discrepancy:**
    - [x] RTP difference persists after standardization (`main.py` ~66%, `optimized.py` ~16.7% with seed 42).
    - [x] Compare the core simulation logic (`optimized.py` imports `run_base_game_spin` etc. from `main.py`).
    - [x] Compare detailed output logs (`diff spins_standard_test_2.csv spins_optimized_test_2_fixed.csv`) - **Confirmed divergence in results despite same seed.**
    - [x] Eliminate parallel RNG state as the *sole* cause (divergence persists with `--cores 1`).
    - [x] Eliminate Numba env vars (`NUMBA_FASTMATH`) as cause (running `main.py` with env var showed no change).
    - [x] Eliminate test mocking interference (Code search shows mocks are locally scoped in tests).

- [ ] **5. Fix Identified Issues:**
    - [x] Fixed incorrect RNG seeding within `optimized.py` workers (did not solve RTP discrepancy).
    - [ ] **Next Steps based on Investigation:**
        - [ ] **Hypothesis:** `main.py` has a bug causing inflated RTP, while `optimized.py` correctly reflects the low RTP from `config.py`.
        - [ ] **Action:** Debug `main.py` execution for a few divergent spins (using `pdb` or print statements) to find where win calculation differs from expectations based on `config.py`.
        - [ ] **Alternative Action:** Modify `config.py` to have non-zero weights for Wilds/Scatters. Rerun standardized tests for `main.py` and `optimized.py`. Do they now produce *similar* RTPs? This helps isolate if the bug relates specifically to the zero-weight handling.
        - [ ] If `main.py` bug is found, fix it and rerun tests.
        - [ ] Revisit performance analysis (Step 3 - profiling) if RTPs are aligned but speed still differs greatly.

- [ ] **6. Final Confirmation:**
    - [ ] Once RTPs align and any bugs are fixed, run the original 500k spin comparison again to confirm consistent results. 