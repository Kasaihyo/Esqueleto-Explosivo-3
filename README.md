# Esqueleto Explosivo 3 Clone

This project is a Python-based simulator for a slot game inspired by Esqueleto Explosivo 3, based on the specifications in the `docs/` folder. It allows for simulating game spins, calculating RTP, analyzing hit frequency, and debugging game mechanics.

## Project Structure

- `simulator/`: Contains the core Python code for the simulation engine.
    - `main.py`: The main entry point for running simulations.
    - `config.py`: Configuration parameters (paytables, probabilities, feature settings).
    - `core/`: Core game logic (Grid, Symbols, State, etc.).
- `tests/`: Contains unit and integration tests for the simulator components (`pytest`).
- `docs/`: Project documentation, including mathematical specifications (`GDD Math.md`), command references (`commands.md`), and other notes.
- `simulation_results/`: Default output directory for simulation summary files and spin logs (CSV). This directory is created automatically.
- `setup.py`: Script for package installation (allows `pip install -e .`).
- `.gitignore`: Specifies intentionally untracked files that Git should ignore.
- `README.md`: This file.

*(Note: `assets/`, `client/`, and `server/` directories might exist but are not directly used by the Python simulator described here.)*

## Setup

1.  **Prerequisites:** Ensure you have Python installed (version 3.9 or later recommended).
2.  **Clone the Repository:**
    ```bash
    git clone <your-repository-url>
    cd <repository-directory>
    ```
3.  **Install Dependencies:** It's recommended to use a virtual environment.
    ```bash
    # Create a virtual environment (optional but recommended)
    python -m venv venv
    # Activate it (Linux/macOS)
    source venv/bin/activate
    # Or (Windows)
    .\venv\Scripts\activate

    # Install the simulator package in editable mode
    pip install -e .
    ```
    This command uses `setup.py` to install the `simulator` package and its dependencies (like `pytest` if listed in `setup.py` or `requirements.txt` - *ensure dependencies are listed there*).

## Running the Simulator

The simulator is run from the command line using `python -m simulator.main`.

**Basic Usage:**

```bash
# Run simulation with default settings from config.py
python -m simulator.main
```

**Common Options:**

```bash
# Run 10,000 spins
python -m simulator.main -n 10000

# Run with verbose output for the first 5 spins
python -m simulator.main -n 100 -v 5

# Run with verbose output only during Free Spins
python -m simulator.main -n 50000 -V

# Run with a specific random seed for reproducibility
python -m simulator.main --seed 42

# Run with enhanced statistics output
python -m simulator.main --enhanced-stats

# Run and save results with a specific ID
python -m simulator.main --id my_test_run
```

For a full list of commands and options, see `docs/commands.md`.

## Running Tests

Unit and integration tests are located in the `tests/` directory and use the `pytest` framework.

**To run all tests:**

```bash
# Ensure you have activated your virtual environment and installed dependencies (pip install -e .)
python -m pytest tests/
```

**To run specific tests or with options:**

```bash
# Run tests with verbose output
python -m pytest -v tests/

# Run tests in a specific file
python -m pytest tests/test_avalanche.py
```

Refer to `docs/commands.md` for more details on running tests.
