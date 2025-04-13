#!/bin/bash
# Specialized script for ROE calculation with different methods

# Default values
SPINS=10000
METHOD="main"  # "main" or "separate"
NUM_SIMS=1000

# Help function
print_help() {
    echo "Usage: ./run_optimized_roe.sh [options]"
    echo ""
    echo "Options:"
    echo "  --spins NUM      Number of spins to simulate (default: 10000)"
    echo "  --method TYPE    ROE calculation method: 'main' or 'separate' (default: main)"
    echo "  --sims NUM       Number of ROE simulations if using separate method (default: 1000)"
    echo "  --help           Display this help message"
    echo ""
    echo "Example:"
    echo "  ./run_optimized_roe.sh --spins 50000 --method main     # Fast ROE using main data"
    echo "  ./run_optimized_roe.sh --spins 50000 --method separate --sims 2000  # Separate ROE sims"
    exit 1
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --spins)
            SPINS="$2"
            shift 2
            ;;
        --method)
            METHOD="$2"
            shift 2
            ;;
        --sims)
            NUM_SIMS="$2"
            shift 2
            ;;
        --help)
            print_help
            ;;
        *)
            echo "Unknown option: $1"
            print_help
            ;;
    esac
done

# Construct command based on parameters
CMD="python run.py --spins $SPINS --enhanced-stats"

# Add ROE-specific parameters
if [ "$METHOD" == "main" ]; then
    CMD="$CMD --roe-use-main-data"
    echo "Using main simulation data for ROE calculation (faster method)"
elif [ "$METHOD" == "separate" ]; then
    CMD="$CMD --roe-separate-sims --roe-num-sims $NUM_SIMS"
    echo "Using separate simulations for ROE calculation ($NUM_SIMS sims)"
else
    echo "Invalid method: $METHOD. Must be 'main' or 'separate'."
    exit 1
fi

echo "Running simulation with $SPINS spins..."
echo "Command: $CMD"
echo ""

# Run the simulation
$CMD

# Display success message
echo ""
echo "ROE calculation complete! Results are available in the simulation_results directory."