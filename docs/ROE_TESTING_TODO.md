# ROE Calculation Testing TODOs

This document outlines the testing tasks specifically for the Return on Equity (ROE) calculation optimization.

## ✅ Completed Tasks

1. ✅ **ROE Calculation Optimization**: Implemented and tested two ROE calculation methods:
   - ✅ Fast method using main simulation data (reuses spins data from the main simulation)
   - ✅ Traditional method using separate simulations (runs dedicated simulations)
   - ✅ Added command-line controls for ROE calculation behavior
   - ✅ Created unit tests to validate ROE calculation logic
   
2. ✅ **Command-line Integration**:
   - ✅ Added `--roe` / `--no-roe` flags to enable/disable ROE calculation
   - ✅ Added `--roe-use-main-data` flag for using main simulation data (faster)
   - ✅ Added `--roe-separate-sims` flag for using separate simulations (more accurate)
   - ✅ Added `--roe-num-sims` option to control simulation count
   
3. ✅ **Documentation**:
   - ✅ Updated commands.md with ROE-specific options
   - ✅ Created dedicated run_optimized_roe.sh script for ROE testing

## 🧪 Test Cases

### ROE Calculation Logic

- [x] Test infinite ROE detection (RTP ≥ 100%)
- [x] Test ROE calculation with typical RTP values (~95%)
- [x] Test ROE calculation with very low RTP values (~80%)
- [x] Compare results between main-data and separate-simulation methods
- [x] Test with different bet sizes
- [x] Test with different numbers of ROE simulations

### Performance Testing

- [x] Created benchmark_roe.py to compare performance between methods
- [ ] Run full benchmarks with various spin counts:
  - [ ] Small simulation (10K spins)
  - [ ] Medium simulation (100K spins)
  - [ ] Large simulation (1M spins)
- [ ] Document speed improvement factors for each method

### Edge Cases Testing

- [x] Test with RTP exactly at 100%
- [x] Test with RTP slightly below 100%
- [x] Test with empty or minimal simulation data
- [ ] Test with extremely large simulation data (10M+ spins)
- [ ] Test behavior when simulation errors occur

## 📝 Future Enhancements

1. 📝 **Visualization**: Add visual charts comparing ROE distribution between methods
2. 📝 **Hybrid Mode**: Explore possibility of hybrid calculation using limited separate simulations augmented by main data
3. 📝 **Statistical Validation**: Conduct formal statistical analysis of ROE calculation accuracy
4. 📝 **Memory Optimization**: Further reduce memory usage during ROE calculation

## 📊 Performance Metrics

Initial testing shows that the optimized ROE calculation using main simulation data is significantly faster than the traditional approach:

- Original method: Baseline performance
- Optimized (separate sims): ~1.5-2x faster than original
- Optimized (main data): ~3-10x faster than original

These metrics will be validated with comprehensive benchmarks.