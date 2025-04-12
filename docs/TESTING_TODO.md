# Simulator Test Cases TODO List

This document tracks the test cases needed to ensure the simulator functions correctly according to the GDD.

**Status:**
*   `TODO`: Not yet implemented.
*   `[DONE - Partial]`: Some tests implemented, but more needed for full coverage.
*   `[DONE]`: All planned tests for this category are implemented.

---

## Test Case Categories

### Grid Initialization (`test_grid_init.py` - *Created*)
*   [✅] Test correct dimensions are set.
*   [✅] Test initial grid population uses correct weights (BG vs FS) - Requires mocking `random.choices`.

### Cluster Detection (`test_clusters.py`)
*   [✅] Basic horizontal/vertical clusters.
*   [✅] Complex L-shaped/T-shaped clusters.
*   [✅] Clusters involving Wilds (W and EW).
*   [✅] Multiple separate clusters in one grid.
*   [✅] Ensure minimum size (5) is enforced (e.g. test cluster size 4 is not found).
*   [✅] Ensure Scatters/Empty don't form clusters (implicitly tested, but explicit test good).

### Pay Calculation (`test_payouts.py`)
*   [✅] Test payouts for various symbols and cluster sizes (min, mid, max).
*   [✅] Test payout for cluster size > 15 uses 15+ value.
*   [✅] Test payout with BG multiplier (1x, 4x, 32x).
*   [✅] Test payout with FS multiplier (various base levels and avalanche counts).
*   [✅] Test zero payout for non-paying symbols (W, EW, S, EMPTY).

### Wild Spawning (`test_spawning.py`)
*   [✅] Test exactly 1 wild spawns per winning cluster.
*   [✅] Test spawn occurs within original cluster footprint.
*   [✅] Test spawn occurs in a location marked for clearing (implicitly tested by footprint check).
*   [✅] Test W/EW probability split (requires mocking `random.random` or statistical test).
*   [✅] Test collision handling: multiple overlapping clusters still result in correct total spawns in valid locations.
*   [✅] Test no spawn if cluster footprint is entirely filled by other spawns/unexploded EWs.

### Explosivo Wild (EW) Explosion (`test_explosions.py`)
*   [✅] Explodes after win calc if present (spawned or landed).
*   [✅] Correct 3x3 area affected.
*   [✅] Correctly destroys only LP symbols.
*   [✅] HP, W, EW, Scatter symbols are not destroyed.
*   [✅] Correctly triggers avalanche check (flag `did_explode` is true even if no LPs destroyed).
*   [✅] Test EW collected exactly once when EW is part of cluster AND explodes.

### Avalanche Mechanic (`test_avalanche.py` - *[DONE]*)
*   [✅] Test simple drop in a single column.
*   [✅] Test drops in multiple columns.
*   [✅] Test symbols dropping past existing symbols.
*   [✅] Test top refill uses correct distribution (BG vs FS) - Requires mocking/statistical test.
*   [✅] Test `landed_coords` correctness after refill (includes drops & refills).
*   [✅] Test empty column handling.
*   [✅] Test no change if grid is full or no symbols move.
*   [✅] Test interaction with spawned wilds (do they fall correctly?).

### Mucho Multiplier (Base Game - `test_sequences.py` - *Created*)
*   [✅] Test starts at 1x.
*   [✅] Test increments only on winning cluster, not EW explosion only.
*   [✅] Test caps at 32x after sufficient wins.
*   [✅] Test resets on the next `initialize_spin`.

### Scatter Trigger (Base Game - `test_sequences.py` - *[DONE]*)
*   [✅] Test no trigger for 0, 1, 2 scatters over sequence.
*   [✅] Test correct initial spins awarded for 3, 4, 5, 6+ scatters over sequence.
*   [✅] Test scatters counted correctly across initial drop and multiple avalanches.

### Free Spins Feature (`test_freespins.py` - *[DONE]*)
*   [✅] Test FS uses FS symbol distribution for refills.
*   [✅] Test FS multiplier uses correct trail based on current base level.
*   [✅] Test FS multiplier uses avalanche count within the *current* free spin.
*   [✅] Test FS multiplier resets to base level at start of each FS spin.
*   [✅] Test remaining spins tracked correctly.
*   [✅] Test Retrigger awards correct spins for 2, 3, 4, 5+ scatters within one FS sequence.
*   [✅] Test EW Collection increments session counter correctly (from clusters & explosions).
*   [✅] Test Upgrade Check: Correctly calculates pending upgrades (0, 1, 2+).
*   [✅] Test Upgrade Application: Base multiplier increases correctly (respecting cap).
*   [✅] Test Upgrade Application: Correct number of extra spins awarded.
*   [✅] Test Upgrade Application: EW session counter reduced correctly.
*   [✅] Test Upgrade applied at start of *next* spin.

### Full Simulation Runs (`test_integration.py` - *[DONE - Partial]*)
*   [✅] Test basic statistics calculation (RTP, Hit Freq, FS Freq).
*   [✅] Test logging output formats (CSV, Summary Txt).
*   [✅] Command-line argument parsing tests are skipped for now, but functional tests have been provided.

### Mucho Multiplier (Free Spins - `test_freespins.py` - *[DONE]*)
*   [✅] Test starts at FS Base Level 1x on entry.
*   [✅] Test uses correct trail based on current FS Base Level (e.g., Base 2x -> 2, 4, 8...).

### Optimized Simulator Tests (`test_optimized.py` - *TODO*)
*   [ ] Test parallel processing produces identical results to sequential processing.
*   [ ] Test batch processing correctly aggregates results.
*   [ ] Test hardware detection correctly identifies Apple Silicon.
*   [ ] Test hardware detection correctly identifies CUDA capabilities.
*   [ ] Test JIT-compiled functions produce correct results.
*   [ ] Test visualization functions correctly generate plots.
*   [ ] Test memory optimization correctly adjusts parameters.
*   [ ] Test error handling for invalid parameter combinations.
*   [ ] Benchmark tests comparing optimized vs original implementation.
*   [ ] Test auto-configuration produces sensible parameters for different hardware profiles.