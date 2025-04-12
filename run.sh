#!/bin/bash
# Simple wrapper script for the auto-optimized simulator

# Check if Python is available
if ! command -v python &> /dev/null; then
    echo "Python is not installed or not in PATH. Please install Python 3.8+"
    exit 1
fi

# Install required dependencies if needed
if ! python -c "import numba" &> /dev/null || ! python -c "import joblib" &> /dev/null; then
    echo "Installing required dependencies..."
    pip install -e . > /dev/null 2>&1
fi

# Check if installation succeeded
if ! python -c "import numba" &> /dev/null; then
    echo "Warning: Numba installation failed. Performance will be degraded."
    echo "Try running: pip install numba"
fi

# Run the optimized script with all arguments passed through
python run.py "$@"

# Check exit code and report results
EXIT_CODE=$?
if [ $EXIT_CODE -eq 0 ]; then
    echo "Simulation completed successfully!"
    echo "Find your results in the simulation_results directory."
    
    # Count result files for this run
    COUNT=$(ls -1 simulation_results/*.csv 2>/dev/null | wc -l)
    if [ $COUNT -gt 0 ]; then
        echo "Generated $COUNT result files."
    fi
else
    echo "Simulation failed with exit code $EXIT_CODE"
    echo "Check the error message above for details."
fi

exit $EXIT_CODE