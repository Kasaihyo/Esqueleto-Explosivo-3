# Simulator Commands

This document lists common commands to run the Python simulator (`simulator/main.py`) from the **project root directory** (`Esqueleto Explosivo 3/`).

Remember to execute these using `python -m simulator.main ...`

## Standard Simulation

Runs the simulation using the default number of spins specified in `simulator/config.py` (currently `TOTAL_SIMULATION_SPINS`) and logs results to timestamped files in `simulation_results/`.

```bash
python -m simulator.main
```

## Simulation with Specific Number of Spins

Runs the simulation for a specific number of spins (e.g., 10,000).

```bash
python -m simulator.main -n 10000
```
(Replace `10000` with the desired number of spins).

## Verbose Simulation (Visual Output)

Runs a small number of spins (e.g., 5) and prints the detailed grid state changes (initial fill and after each avalanche) to the console. Useful for visual validation.

```bash
python -m simulator.main -n 5 -v 5
```
(The first `-n 5` sets the total spins, the second `-v 5` makes those 5 spins verbose).

## Simulation with Specific Bet Amount

Runs the simulation using a different base bet amount (e.g., 0.5).

```bash
python -m simulator.main -b 0.5
```

## Reproducible Simulation (Using a Seed)

Runs the simulation using a specific random seed (e.g., 42) to ensure the results are the same every time you run it with that seed. Useful for debugging or comparing changes.

```bash
python -m simulator.main --seed 42
```
(Combine with `-n` for a specific spin count if needed).

## Simulation with Custom Run ID

Runs the simulation and names the output log files using a specific ID instead of just a timestamp.

```bash
python -m simulator.main --id my_first_tuning_run
```

## Enhanced Statistics Mode

Runs the simulation and displays enhanced statistics including win distribution ranges, max win details, and visualizations of the top 10 wins.

```bash
python -m simulator.main --enhanced-stats
```

## Statistics-Only Mode

Runs the simulation with no verbose output, but shows detailed statistics including win ranges and top wins visualization. This is useful for quick analysis of slot performance metrics.

```bash
python -m simulator.main --stats-only
```

For example, run 10,000 spins with a consistent seed and view detailed statistics:

```bash
python -m simulator.main -n 10000 --seed 123 --stats-only
```

## Verbose Free Spins Only

Runs the simulation silently until a Free Spins feature is triggered, then prints the detailed grid updates *only* for the duration of that Free Spins feature. This is useful for debugging the Free Spins logic without seeing verbose output for the base game spins leading up to it.

*Note: This overrides the standard `-v` flag if both are used.*

```bash
# Run until first FS trigger, then show FS details
python -m simulator.main -V

# Run for a specific number of spins, showing details for ALL FS features triggered
python -m simulator.main -n 50000 -V
```

## Combined Examples

Runs 50,000 spins, with the first 10 base game spins being verbose (using `-v`), using a seed of 123, a base bet of 2.0, and a custom run ID.

```bash
python -m simulator.main -n 50000 -v 10 --seed 123 -b 2.0 --id tuning_v0.1
```

Runs 50,000 spins, showing verbose output *only* during any triggered Free Spins feature (using `-V`), with a seed and custom ID.

```bash
python -m simulator.main -n 50000 -V --seed 456 --id fs_debug_run
```

Runs 100,000 spins for RTP testing with statistics-only mode and saves with a descriptive ID:

```bash
python -m simulator.main -n 100000 --stats-only --id rtp_analysis_v1
```

Runs a smaller simulation with detailed verbose output for the first 3 spins and enhanced statistics:

```bash
python -m simulator.main -n 1000 -v 3 --enhanced-stats --id detailed_analysis
``` 

## Running Unit Tests

To verify the core logic of the simulator using the implemented unit tests, navigate to the project root directory (`Esqueleto Explosivo 3`) and run one of the following commands:

### Using pytest (Recommended)

```bash
# Run all tests
python -m pytest tests/

# Run a specific test file
python -m pytest tests/test_avalanche.py

# Run a specific test class or method
python -m pytest tests/test_avalanche.py::TestAvalancheMechanic
python -m pytest tests/test_avalanche.py::TestAvalancheMechanic::test_simple_drop_single_column

# Run tests with verbose output
python -m pytest -v tests/

# Run tests and stop on first failure
python -m pytest -xvs tests/
```

### Using unittest (Alternative)

```bash
# Run all tests
python -m unittest discover tests

# Run a specific test file
python -m unittest tests/test_avalanche.py

# Run a specific test class
python -m unittest tests.test_avalanche.TestAvalancheMechanic
```

### Setup for Running Tests

Before running the tests, make sure you have installed the simulator package in development mode:

```bash
# From the project root directory
pip install -e .
```

This ensures that imports in the test files can correctly resolve the simulator module.

### Interpreting the Output

*   **`.` (dot):** Each test that passed successfully.
*   **`F`:** Indicates a test that failed (an assertion was false).
*   **`E`:** Indicates a test that encountered an error (an unexpected exception occurred).
*   **`s`:** Indicates a test that was skipped.

After running the tests, a summary line will appear (e.g., `50 passed in 0.08s`).

*   **If all tests pass:** You will see a summary of passed tests.
*   **If tests fail or have errors:** You will see details about each failure, including which test failed, where the failure occurred in the code, and the reason for the failure (e.g., `AssertionError: expected != actual`).