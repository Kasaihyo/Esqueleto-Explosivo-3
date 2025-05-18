# RNG Seeding Strategy

This document describes how **Esqueleto Explosivo 3** guarantees *deterministic* output while still supporting massive parallel simulations.

## 1.  Why determinism?
Auditors and mathematicians must be able to reproduce every Return-To-Player (RTP) figure that the engine reports.  A fully deterministic pipeline ensures that a run can be re-executed months later, on different hardware, and produce byte-identical statistics.

## 2.  `SpinRNG`
All randomness is channelled through a lightweight wrapper around `random.Random` located at `simulator/core/rng.py`.

```python
from simulator.core.rng import SpinRNG
rng = SpinRNG(seed=42)
value = rng.random()
```

The wrapper exposes only the subset of functionality used by the simulator (currently `random()`, `randint()`, and `choices()`), ensuring that new, uncontrolled calls do not slip in unnoticed.

## 3.  Per-worker streams
Whenever we spawn parallel workers (e.g. during ROE calculations) each process receives its own `SpinRNG` instance seeded as:

```
seed_worker_i = base_seed + i
```

where `base_seed` originates from the CLI's `--seed` argument.  If the user does **not** supply a seed we fall back to Python's default behaviour which already incorporates high-entropy OS sources.

## 4.  CLI flags
`run.py` exposes a `--seed` option that forwards the number to the core simulator.  Running twice with the same flag guarantees identical CSV/summary output.

```
python run.py --seed 1234   # results A
python run.py --seed 1234   # results B   <- A == B
```

If the flag is omitted the run is intentionally *non-deterministic* to increase test coverage of different symbol sequences.

## 5.  Verifying determinism
A regression test will spin 10 000 rounds twice—using the same fixed seed—and assert that all high-level statistics (RTP, hit-rate, etc.) match within an extremely tight tolerance.

```bash
pytest tests/test_regression.py
```

## 6.  Future work
* Extend `SpinRNG` with NumPy's `Generator` interface once the simulation hot-path moves to vectorised code.
* Ensure *all* stray `random.*` calls are routed through the injected RNG instance. 