# Esqueleto Explosivo 3 Simulator

This directory contains the Python simulation code for the Esqueleto Explosivo 3 game, including both the standard implementation and an optimized version with parallel processing.

## Project Structure

- `main.py`: Original simulation entry point
- `optimized.py`: Parallelized, optimized implementation with hardware acceleration
- `config.py`: Game parameters and configurations
- `core/`: Core simulation logic (grid, symbols, mechanics)
- `requirements.txt`: Python dependencies

## Simulator Versions

### Standard Simulator
The standard simulator in `main.py` processes spins sequentially and is suitable for smaller simulation runs or environments with limited resources.

### Optimized Simulator
The optimized simulator in `optimized.py` provides:
- Multi-core parallel processing with joblib
- GPU acceleration via Numba where available
- Automatic hardware detection and configuration
- Batch processing for improved memory usage
- Visualization capabilities for win distributions

## Installation

Install the simulator dependencies with:

```bash
pip install -r requirements.txt
```

For the optimized simulator, additional dependencies are required:

```bash
pip install joblib numba matplotlib psutil pyarrow tqdm
```

## Usage

### Auto-Optimized Mode (Recommended)

For the simplest experience, use the provided shell script which automatically detects your hardware and sets optimal parameters:

```bash
./run.sh
```

Or with custom parameters:

```bash
./run.sh --spins 1000000 --bet 2.0
```

### Manual Mode

Run the standard simulator:

```bash
python -m simulator.main --spins 1000000 --bet 2.0
```

Run the optimized simulator:

```bash
python -m simulator.optimized --spins 1000000 --bet 2.0 --batch-size 1000 --cores 8
```

## Testing and Benchmarking

### Running Tests

Run all tests:

```bash
python -m unittest discover tests
```

Run specific test suite:

```bash
python -m unittest tests.test_optimized
```

### Benchmarking

Compare performance between standard and optimized implementations:

```bash
python tests/benchmark.py
```

With custom parameters:

```bash
python tests/benchmark.py --spins 100000 --batch-sizes 100 500 1000 --cores 4 8
```

The benchmark script generates performance comparison plots in the `simulation_results/` directory.

## Hardware Optimization Notes

The optimized simulator automatically detects and utilizes:

- Apple Silicon ARM processors: Optimized for M1/M2/M3 chips
- CUDA-capable NVIDIA GPUs: Uses GPU acceleration where available
- Multi-core CPUs: Scales processing across available cores
- Available system memory: Adjusts batch sizes based on RAM

## Documentation

For detailed descriptions of command options, see `docs/commands.md`.