import pytest
from simulator.main import calculate_retrigger_spins

@ pytest.mark.parametrize("scatter_count, expected", [
    (0, 0),  # Below trigger threshold
    (1, 0),  # Below trigger threshold
    (2, 3),  # Minimum retrigger
    (3, 5),  # Standard retrigger
    (4, 7),  # Maximum defined retrigger
    (5, 9),  # One above max defined
    (6, 11), # Two above max defined
    (10, 7 + (10 - 4) * 2),  # Well above max defined
])
def test_calculate_retrigger_spins(scatter_count, expected):
    """Test that calculate_retrigger_spins returns correct extra spins for various scatter counts."""
    assert calculate_retrigger_spins(scatter_count) == expected 