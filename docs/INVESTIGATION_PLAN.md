# RTP Discrepancy Investigation Plan

## Overview
This document outlines a systematic investigation into the significant RTP (Return to Player) differences between the regular simulator (main.py) and optimized simulator (optimized.py) implementations.

## Test Results Summary (2025-04-20)

| Test ID | Version | Seed | RTP | Hit Frequency | FS Triggers | Max Win |
|---------|---------|------|-----|---------------|-------------|---------|
| regular_50k_seed1 | Regular | 111 | 38.98% | 18.60% | 0.144% | 187.80x |
| optimized_50k_seed1 | Optimized | 111 | 17.91% | 18.98% | 0.082% | 25.00x |
| regular_50k_seed2 | Regular | 222 | 38.83% | 18.75% | 0.186% | 188.30x |
| optimized_50k_seed2 | Optimized | 222 | 17.60% | 18.85% | 0.090% | 15.40x |
| regular_50k_seed3 | Regular | 333 | 38.75% | 18.84% | 0.168% | 156.50x |
| optimized_50k_seed3 | Optimized | 333 | 18.08% | 18.79% | 0.126% | 25.60x |

## Key Discrepancies
- RTP consistently ~20 percentage points higher in regular version
- Similar hit frequencies between versions
- Free spins trigger nearly twice as often in regular version
- Max win values substantially higher in regular version

## Investigation Tasks

### 1. Win Calculation Analysis ✅
- [x] Verified both versions use identical win calculation logic (via grid.calculate_win)
- [x] Confirmed both use the same configuration values (PAYTABLE and symbol weights)
- [x] Observed results diverge immediately even with the same seed
- [x] Win values differ significantly between versions with same seed: 
  - Regular: Spin 2 win = 5.5
  - Optimized: Spin 2 win = 0.5, Spin 3 win = 1.0

### 2. Free Spins Analysis
- [ ] Add detailed FS trigger and win tracking
- [ ] Compare FS retrigger rates between versions
- [ ] Analyze multiplier progression during FS
- [ ] Calculate contribution of FS to overall RTP

### 3. Symbol Generation Analysis
- [ ] Verify symbol frequencies match configuration
- [ ] Compare wild symbol generation between versions
- [ ] Analyze scatter distribution patterns
- [ ] Check for differences in avalanche behavior

### 4. Multiplier Handling
- [ ] Track multiplier values for each win
- [ ] Verify multiplier increment logic
- [ ] Compare avalanche sequence lengths
- [ ] Check multiplier reset conditions

### 5. Cluster Detection
- [ ] Add logging for cluster sizes and compositions
- [ ] Compare cluster detection results between versions
- [ ] Analyze differences in adjacent symbols detection
- [ ] Check for edge cases in grid boundaries

### 6. Technical Analysis
- [ ] Review all imported functions and dependencies
- [ ] Check for numerical precision issues
- [ ] Compare RNG sequence patterns
- [ ] Analyze memory usage and garbage collection effects

## Action Plan

1. **Initial Setup**
   - [ ] Implement detailed logging in both versions
   - [ ] Create test harness for side-by-side comparison

2. **Data Collection**
   - [ ] Run both versions with same seeds (111, 222, 333)
   - [ ] Collect detailed logs for at least 1,000 spins
   - [ ] Store all simulation data for analysis

3. **Comparative Analysis**
   - [ ] Identify first spin where results diverge
   - [ ] Trace execution through critical components
   - [ ] Identify specific cause of RTP differences

4. **Hypotheses Testing**
   - [ ] Test fix for multiplier handling
   - [ ] Test fix for FS trigger logic
   - [ ] Test fix for win calculation

5. **Verification**
   - [ ] Run full tests with proposed fixes
   - [ ] Verify RTP convergence between versions
   - [ ] Maintain performance advantages in optimized version

## Success Criteria
- Both versions produce similar RTP values (within 1%)
- Both versions maintain similar hit frequencies and FS trigger rates
- Optimized version maintains performance advantages

## Reporting
- Document all findings and root causes
- Provide detailed fix recommendations
- Update code documentation