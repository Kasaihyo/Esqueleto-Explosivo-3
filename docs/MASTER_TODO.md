# 📌  MASTER TODO – RTP Calibration Engine

This single file is **the only backlog** for the repository.  No other `TODO*.md` files or `# TODO:` comments should exist; add new work here.

---

## Epic 1 – Consolidate a single math engine
| ID | Task | Acceptance Criteria |
|----|------|--------------------|
| E1-S1 | ✅ Delete `simulator/optimized.py` and any duplicate helper code. | File removed; `pytest -q` green; no stray imports. |
| E1-S2 | ✅ Remove `server/` & `client/` (visualiser migrates to separate repo). | Folders gone; README & setup adjusted; CI passes. |
| E1-S3 | ✅ Ensure all avalanche/cluster/payout logic lives in `simulator/core/`. | Search confirms one definition per logical routine. |
| E1-S4 | ✅ Update documentation (`README`, architecture diagram) to reflect single-engine structure. | Docs render without references to removed files. |
| E1-S5 | ✅ Delete obsolete tests that referenced `optimized.py` or web server. | `pytest -q` passes with same coverage %. |

## Epic 2 – Deterministic RNG
| ID | Task | Acceptance Criteria |
|----|------|--------------------|
| E2-S1 | ✅ Implement `SpinRNG` wrapper around `random.Random`. | Unit test proves sequence reproducibility. |
| E2-S2 | ✅ Inject RNG instance everywhere `generate_random_symbol` is used. | `grep -R "random.choices("` matches only RNG helper. |
| E2-S3 | ✅ Spawn one RNG per worker in parallel runs (`seed = base + id`). | Two identical runs produce byte-identical stats. |
| E2-S4 | ✅ Expose `--seed` CLI flag in `run.py` & optimizer to set global seed. | Passing flag reproduces previous results exactly. |
| E2-S5 | ✅ Add `docs/RNG.md` describing seeding strategy for audit trail. | Document exists & reviewed. |

## Epic 3 – Single source-of-truth config
| ID | Task | Acceptance Criteria |
|----|------|--------------------|
| E3-S1 | ✅ Delete fallback symbol / multiplier dicts from removed code. | Only `simulator/config.py` defines weights. |
| E3-S2 | ✅ Store weights as `numpy.ndarray` for vector maths. | Type check passes; all references updated. |
| E3-S3 | ✅ Add `validate_config()`; call at simulator import. | Raises on invalid sums; tested. |
| E3-S4 | ✅ Provide `config.save_weights("weights_YYYYMMDD.json")` utility. | File round-trips back into numpy arrays identically. |
| E3-S5 | ✅ Document weight format & validation rules in `docs/CONFIG.md`. | Auditors can follow steps to reproduce validation. |

## Epic 4 – Weight-optimiser CLI
| ID | Task | Acceptance Criteria |
|----|------|--------------------|
| E4-S1 | ✅ Create `tools/weight_optimizer.py` using differential evolution. | `python -m tools.weight_optimizer --target 94.5` outputs weights. |
| E4-S2 | ✅ Support `--out weights.json` to save weight blob + metadata. | File written, schema documented. |
| E4-S3 | ✅ Add option `--resume` that starts search from a previous JSON. | Optimizer continues and improves objective. |
| E4-S4 | ✅ Parallelise candidate evaluation with `multiprocessing` respecting RNG isolation. | Runtime scales ≥ 0.8× cores. |
| E4-S5 | ✅ Generate convergence plot (`png`) when `--plot` flag provided. | File saved; opens without error. |

## Epic 5 – Performance & cleanliness
| ID | Task | Acceptance Criteria |
|----|------|--------------------|
| E5-S1 | ✅ Replace symbol objects in hot path with `int8` ids + lookup. | RTP delta ≤ 0.01 %; ≥ 5× speed-up 1-core. |
| E5-S2 | ✅ Optional Numba `@njit` cluster detection (`--jit` flag). | Python & jit paths return identical clusters. |
| E5-S3 | ✅ Drop unguarded `print`; use `logging` with `--verbose`. | CI regex check finds no bare prints. |
| E5-S4 | ✅ Add memory-profiler benchmark; ensure < 1 GB at 10 M spins. | Benchmark script passes. |
| E5-S5 | ✅ Run `black`, `isort`, `flake8` in pre-commit hook. | `pre-commit run --all-files` passes. |

## Epic 6 – Regression & CI guardrails
| ID | Task | Acceptance Criteria |
|----|------|--------------------|
| E6-S1 | ✅ Add `tests/test_regression.py` (10 000 spins, fixed seed, RTP tolerance ±0.05 %). | Test fails on accidental math drift. |
| E6-S2 | ✅ GitHub Actions workflow: lint, tests, coverage ≥ 90 %. | PR blocked if requirements unmet. |
| E6-S3 | ✅ Integrate `codecov` badge in README. | Badge shows latest coverage %. |
| E6-S4 | ✅ Add `mypy` type-checking to CI. | `mypy .` exits 0. |
| E6-S5 | ✅ Add `docs/CHANGELOG.md`; require entry in each PR via CI check. | CI fails if PR lacks changelog entry. |

---

✋ **Process:**
1. Open PR implementing one sub-task.
2. Link the sub-task ID in PR title (e.g. `E2-S2: inject RNG`).
3. Mark finished row with ✅.

When all subtasks are complete the engine will be lean, deterministic and ready for rapid symbol-weight tuning. 