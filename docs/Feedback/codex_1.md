Here’s a high‐level pass over the codebase. I’ve grouped my findings into “Bugs / Incorrect or Incomplete Behavior” and “Abnormalities / Code Smells / Dead-code” to make it easier
 to scan.

    1. Bugs / Incorrect or Incomplete Behavior

 • Free-Spins CSV logging is incomplete
   – In run_simulation(), the CSV columns “FS_Spins_Played”, “FS_EW_Collected” and “FS_Retrigger_Spins_Awarded” are written as placeholders (always 0).
   – run_free_spins_feature() only returns total FS win, never the per-spin FS stats needed to populate those columns.

 • Duplicate‐but‐un‐used FS / BG trigger logic
   – There’s a full GameState.finalize_spin_sequence() implementation of trigger/retrigger/upgrades, but main.py never calls it. Instead main.py re-implements all of that inline.
   – Risk: if one gets tweaked and the other doesn’t, BG/FS trigger‐retrigger rules will drift out of sync.

 • Dead helper never used
   – Grid._process_column_gravity() is defined but never called; apply_avalanche() re-implements the same logic inlined.

 • Unused configuration constant
   – config.PROB_SPAWN_E_WILD is never referenced; Grid.process_explosions_and_spawns() just does
        if rng.random() < PROB_SPAWN_WILD: spawn WILD else E_WILD
     which works only because PROB_SPAWN_WILD + PROB_SPAWN_E_WILD == 1.0 today.  If you ever retune those two you’ll silently break the E_WILD ratio.

 • Hard-coded Avalanche cap
   – Both run_base_game_spin() and run_free_spins_feature() hard-code a 50-avalanche cap.  That too should be a constant, not a magic “50” sprinkled in two places.

 • JIT stub never actually accelerates
   – generate_random_symbol() takes an unused sim_type argument (“main” vs “optimized”), but it never switches on it to pick the numpy‐vectorized path.

 • calculate_roe() won’t run without extra deps
   – It does an in‐function “from joblib import Parallel…”, but if joblib (and its peer tqdm) aren’t installed you’ll get runtime errors.  setup.py does list joblib as a
dependency, but if someone installs only numpy / core simulator that hole will blow up only at calculate_roe time.

    1. Abnormalities / Code Smells / Dead-code

 • GameState is almost entirely ignored
   – Aside from being passed into Grid.apply_avalanche() as a signal to pick BG vs FS weights, none of its methods (update_after_clusters, accumulate_scatter,
finalize_spin_sequence, etc.) are ever used.

 • Typos & odd aliases in config
   – The alias FS_MUCHO_MULTIPLIER_TRAIL looks like a typo of “FS_MULTIPLIER”…
   – There sare a number of BETPLUS_* and FEATURE_BUY_* constants in config, but the code has no BetPlus or Feature-Buy logic.

 • Unused / commented-out imports everywhere
   – Numerous “# import copy  # F401 unused” and similar lines clutter main.py and run.py.

 • Logger setup is messy
   – main.py adds a handler at import time if no handlers exist, then in main it clears and re-adds handlers.  If you import run_simulation() into another app you’ll get odd
logging state.

 • Direct use of private Grid._set_symbol and Grid._get_symbol in tests
   – Tests drive and inspect very internal bits of Grid rather than using a stable public API.  That makes refactoring dangerous.

 • Duplicate cluster‐merge logic
   – find_clusters_python() builds clusters and then merges by base_symbol.name itself, rather than taking advantage of a single BFS over all connected (symbol + wild) positions.


Recommendations

    1. Pull all of the FS/BG‐trigger, retrigger, upgrade logic into GameState and have main.py just call `state.finalize_spin_sequence()` so there’s a single source of truth.
    2. Flesh out run_free_spins_feature() so it returns the per-spin details (spins_played, EWs_collected, retriggers_awarded) and wire those back into the CSV.
    3. Remove or actually wire in Grid._process_column_gravity(), and delete the dead argument `sim_type` from generate_random_symbol() (or fully implement the “optimized”
branch).
    4. Expose the avalanche cap (50) as a named constant and centralize it, rather than embedding “50” twice.
    5. Fix the E_WILD spawn logic to consume both PROB_SPAWN_WILD and PROB_SPAWN_E_WILD so changing either in config actually takes effect.
    6. Prune out the BetPlus/FeatureBuy constants (or implement the feature!) and clean up the commented-out imports.
    7. Consolidate logger configuration (do it once, in a single place) so that importing main.run_simulation() doesn’t add multiple handlers or clear user handlers.
    8. Consider giving Symbol an `id` attribute at construction (from config.SYMBOL_TO_ID) so you don’t have to round-trip through getattr(.., "id") in Grid._set_symbol.
