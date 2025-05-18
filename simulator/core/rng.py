import random
from typing import Any, List


class SpinRNG:
    """Lightweight deterministic RNG wrapper used by the simulator.

    The class is a very thin proxy around :class:`random.Random` but exposes
    only the handful of methods that are required by the simulator codebase.
    A dedicated object – rather than the implicit global ``random`` module –
    lets us inject a unique RNG stream into each worker or game instance and
    therefore guarantee *reproducible* sequences while still being able to
    run simulations in parallel.

    Parameters
    ----------
    seed
        Optional seed to initialise the generator.  If *None* the underlying
        RNG will follow CPython's default seeding behaviour (normally based on
        system entropy).
    """

    def __init__(self, seed: int | None = None):
        self._rng = random.Random(seed)

    # ---------------------------------------------------------------------
    # Public proxy helpers – add more as the simulator codebase requires them
    # ---------------------------------------------------------------------
    def seed(self, seed: int | None = None) -> None:
        """Reseed the generator."""
        self._rng.seed(seed)

    def random(self) -> float:
        """Return the next float in the open interval [0.0, 1.0)."""
        return self._rng.random()

    def randint(self, a: int, b: int) -> int:
        """Return a random integer N such that *a* <= N <= *b*."""
        return self._rng.randint(a, b)

    def choices(
        self, population: list[Any], *, weights: List[float] | None = None, k: int = 1
    ):
        """Thin wrapper around :pymeth:`random.Random.choices`.  The signature is
        intentionally restricted to the arguments we actually use.
        """
        return self._rng.choices(population, weights=weights, k=k)

    def choice(self, seq: list[Any]):
        """Thin wrapper around :pymeth:`random.Random.choice`."""
        return self._rng.choice(seq)

    def getstate(self):  # pragma: no cover – helper only used by debugging tools
        return self._rng.getstate()

    def setstate(self, state):  # pragma: no cover
        self._rng.setstate(state)

    # ------------------------------------------------------------------
    # Convenience dunder helpers so *SpinRNG* behaves like the plain module
    # ------------------------------------------------------------------
    def __getattr__(self, item):  # pragma: no cover – fall-back for rarely used attrs
        """Delegate unknown attributes to the underlying *random.Random* instance."""
        return getattr(self._rng, item)

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"SpinRNG(id={id(self)}, state_hash={hash(str(self._rng.getstate())[:64])})"
        )
