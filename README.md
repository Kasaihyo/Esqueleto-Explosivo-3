# Esqueleto Explosivo 3 Simulator

This project is a Python-based simulator for a slot game inspired by Esqueleto Explosivo 3, based on the specifications in the `docs/` folder. It allows for simulating game spins, calculating RTP, analyzing hit frequency, and debugging game mechanics.

## Project Structure

- `simulator/`: Contains the core Python code for the simulation engine.
    - `main.py`: The original sequential simulation engine.
    - `optimized.py`: The parallelized, hardware-accelerated simulation engine.
    - `config.py`: Configuration parameters (paytables, probabilities, feature settings).
    - `core/`: Core game logic (Grid, Symbols, State, etc.).
- `tests/`: Contains unit and integration tests for the simulator components.
- `docs/`: Project documentation, including mathematical specifications (`GDD Math.md`), command references (`commands.md`), and other notes.
- `simulation_results/`: Default output directory for simulation summary files and spin logs (CSV). This directory is created automatically.
- `run.py`: Automatic hardware detection and optimal configuration script.
- `run.sh`: Simple shell script for running the auto-optimized simulator.
- `setup.py`: Script for package installation (allows `pip install -e .`).
- `.gitignore`: Specifies intentionally untracked files that Git should ignore.
- `README.md`: This file.

## Quick Start

For the simplest experience with optimal performance:

```bash
# Make the script executable (if needed)
chmod +x run.sh

# Run the auto-optimized simulator
./run.sh
```

The script will automatically:
1. Install required dependencies if needed
2. Detect your hardware capabilities (CPU cores, memory, GPU)
3. Configure optimal parameters for your system
4. Run the simulator with these optimized settings

## Manual Setup

1. **Prerequisites:** Ensure you have Python installed (version 3.8 or later recommended).
2. **Clone the Repository:**
   ```bash
   git clone <your-repository-url>
   cd <repository-directory>
   ```
3. **Install Dependencies:** It's recommended to use a virtual environment.
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

## Running the Simulator

### Auto-Optimized Simulator (Recommended)

```bash
# Run with default settings (auto-detects hardware)
./run.sh

# Run with custom parameters
./run.sh --spins 1000000 --bet 2.0
```

### Standard Simulator

```bash
# Run original sequential implementation
python -m simulator.main -n 10000

# Run with verbose output for the first 5 spins
python -m simulator.main -n 100 -v 5

# Run with verbose output only during Free Spins
python -m simulator.main -n 50000 -V
```

### Optimized Simulator (Manual Configuration)

```bash
# Run with specific batch size and core count
python -m simulator.optimized --spins 1000000 --batch-size 500 --cores 8

# Run with visualization enabled
python -m simulator.optimized --spins 1000000 --create-plots
```

For a full list of commands and options, see `docs/commands.md` or `simulator/README.md`.

## Hardware Optimization

The optimized simulator supports:
- **Multi-core CPUs**: Parallelizes work across all available cores
- **Apple Silicon**: Optimized for M1/M2/M3 ARM processors
- **NVIDIA GPUs**: Uses CUDA acceleration where available
- **Memory Optimization**: Adjusts batch size based on system memory

## Testing and Benchmarking

### Run All Tests

```bash
python -m unittest discover tests
```

### Run Benchmark Comparison

```bash
python tests/benchmark.py
```

The benchmark generates performance comparison plots in the `simulation_results/` directory, showing speedup across different configurations.

## Documentation

- `docs/commands.md`: Detailed command usage for all simulator versions
- `docs/GDD Math.md`: Mathematical specifications for the game
- `docs/TESTING_TODO.md`: Test coverage and validation requirements
- `simulator/README.md`: Technical details about the simulator implementation