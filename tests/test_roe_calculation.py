import pytest
from simulator.main import calculate_roe
from simulator.optimized import calculate_roe_optimized

@ pytest.mark.parametrize("rtp", [100.0, 150.0, 1000.0])
def test_main_calculate_roe_infinite(rtp):
    """ROE should be infinite when RTP >= 100%"""
    median, avg = calculate_roe(rtp, base_bet_for_sim=1.0)
    assert median == "Infinite" and avg == "Infinite"

@ pytest.mark.parametrize("num_roe_sims, max_spins", [
    (0, 10),  # no simulations
    (10, 0),  # zero max spins
])
def test_main_calculate_roe_error(num_roe_sims, max_spins):
    """ROE should return Error/Error when no valid simulations are run"""
    median, avg = calculate_roe(50.0, base_bet_for_sim=1.0, roe_bet=1.0, num_roe_sims=num_roe_sims, max_roe_spins=max_spins)
    assert median == "Error" and avg == "Error"

@ pytest.mark.parametrize("rtp", [100.0, 200.0])
def test_optimized_calculate_roe_infinite_main_data(rtp):
    """Optimized ROE should be infinite when RTP >= 100%"""
    median, avg = calculate_roe_optimized(main_simulation_data=[{'total_win': 0}], rtp=rtp, base_bet_for_sim=1.0, roe_bet=1.0, num_roe_sims=1, max_roe_spins=10, roe_cores=1, use_main_data=True)
    assert median == "Infinite" and avg == "Infinite"

@ pytest.mark.parametrize("data_len", [0, 10, 50])
def test_optimized_calculate_roe_error_main_data(data_len):
    """Optimized ROE returns Error/Error if main data length < 100"""
    data = [{'total_win': 1.0}] * data_len
    median, avg = calculate_roe_optimized(main_simulation_data=data, rtp=50.0, base_bet_for_sim=1.0, roe_bet=1.0, num_roe_sims=1, max_roe_spins=10, roe_cores=1, use_main_data=True)
    assert median == "Error" and avg == "Error" 