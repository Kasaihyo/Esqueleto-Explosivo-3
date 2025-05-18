import pytest

from simulator.core.rng import SpinRNG


def test_spin_rng_reproducibility():
    seed = 12345
    rng1 = SpinRNG(seed)
    rng2 = SpinRNG(seed)

    seq1 = [rng1.random() for _ in range(10)]
    seq2 = [rng2.random() for _ in range(10)]

    assert seq1 == seq2, "SpinRNG with same seed should produce identical sequences"


def test_spin_rng_independence():
    rng1 = SpinRNG(1)
    rng2 = SpinRNG(2)

    values1 = [rng1.random() for _ in range(5)]
    values2 = [rng2.random() for _ in range(5)]

    assert values1 != values2, "SpinRNG instances seeded differently should diverge"
