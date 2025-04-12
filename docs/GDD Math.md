# Esqueleto Explosivo 3 Math Clone

## Introduction

This document outlines the product requirements for "Esqueleto Explosivo 3 Clone" a high-volatility video slot game. The game utilizes a 5x5 grid with a Cluster Pays mechanic, featuring Avalanches, Explosivo Wilds, a progressive Mucho Multiplier, a distinct Free Spins round with multiplier upgrades, and optional Bet+/Feature Buy mechanics.

  

## Features & Mechanics

### 1) Core Game Setup

*   **Grid:** 5 reels, 5 rows (5x5).
*   **Pay Mechanic:** Cluster Pays. A win occurs when 5 or more identical symbols land connected horizontally and/or vertically.
*   **Avalanche Mechanic:**
    1. Winning clusters pay according to the paytable.
    2. Symbols in winning clusters are removed.
    3. Wilds may spawn in vacated positions (see 4.7 Wild Spawning).
    4. Explosivo Wilds that landed may explode (see 4.6 Explosivo Wild).
    5. Remaining symbols fall vertically.
    6. Empty positions are filled with new symbols dropping from above, using the active symbol distribution.
    7. This process repeats as long as new winning clusters are formed or Explosivo Wilds explode.

### 2) Symbols

*   **High-Paying (HP):**
    *   Symbol 1: Lady Skull (1 unique symbol)
*   **Low-Paying (LP):**
    *   Symbol 2: Pink Skull
    *   Symbol 3: Green Skull
    *   Symbol 4: Blue Skull
    *   Symbol 5: Orange Skull
    *   Symbol 6: Cyan Skull (5 unique symbols)
*   **Wild (W):**
    *   Standard Wild (Skull with sunglasses "W").
    *   Substitutes for all HP and LP symbols.
    *   Can appear naturally or be spawned.
*   **Explosivo Wild (EW):**
    *   Special Wild (distinct visual, potentially similar base to W).
    *   Substitutes for all HP and LP symbols.
    *   Has an additional explosion feature (see 4.6).
    *   Can appear naturally or be spawned.
*   **Scatter (S):**
    *   Silver Robot Skull.
    *   Triggers Free Spins.
    *   Does not have a direct payout value.
    *   Does not participate in cluster wins.

### 3) Paytable

*   Payouts are awarded for clusters of 5 or more identical symbols, multiplied by the total bet.
*   The following table provides the _initial_ payout structure (subject to tuning via simulation):

| Symbol | Cluster 5 | Cluster 6 | Cluster 7 | Cluster 8-9 | Cluster 10-11 | Cluster 12-14 | Cluster 15+ | Type |
| ---| ---| ---| ---| ---| ---| ---| ---| --- |
| Symbol 1 - Lady Skulls | 1.0x | 1.5x | 2.5x | 5.0x | 7.5x | 25.0x | 150.0x | HP |
| Symbol 2 - Pink | 0.5x | 0.7x | 1.0x | 1.7x | 2.5x | 7.5x | 50.0x | LP |
| Symbol 3 - Green | 0.4x | 0.7x | 0.8x | 1.4x | 2.0x | 6.0x | 40.0x | LP |
| Symbol 4 - Blue | 0.3x | 0.5x | 0.6x | 1.0x | 1.5x | 5.0x | 30.0x | LP |
| Symbol 5 - Orange | 0.3x | 0.4x | 0.5x | 0.8x | 1.2x | 4.0x | 25.0x | LP |
| Symbol 6 - Cyan | 0.2x | 0.3x | 0.4x | 0.6x | 1.0x | 3.0x | 20.0x | LP |

### 4) Symbol Distribution

*   The probability of each symbol appearing in an empty grid position is defined by distribution tables.
*   **Separate distributions** will be used for the Base Game (BG) and Free Spins (FS).
    *   `P_BG(Symbol)`: Probabilities for HP, LPs, W, EW, S during the Base Game.
    *   `P_FS(Symbol)`: Probabilities for HP, LPs, W, EW, S during Free Spins. _Expected to be different from BG, potentially richer in W/EW._
*   The active distribution (BG or FS) applies to both the initial grid population and all subsequent Avalanche refills within that game state.
*   _Note: Specific probabilities are determined and tuned via mathematical simulation._

### 5) Mucho Multiplier

*   A win multiplier applied to cluster wins.
*   **Base Game:**
    *   Starts at **1x** on each new paid spin.
    *   Increases by one step for each Avalanche triggered by a **winning cluster**.
    *   Sequence (6 steps): **1x → 2x → 4x → 8x → 16x → 32x (Max)**.
    *   Stops increasing at 32x within a single spin sequence.
    *   Resets to 1x at the start of the next paid spin.
*   **Free Spins (Enhanced):**
    *   Uses a persistent **FS Base Multiplier**, starting at 1x when FS begins. Max Base Multiplier is 32x.
    *   The current FS Base Multiplier determines the **Active Multiplier Trail** (always 6 steps):
        *   Base 1x -> Trail: 1x, 2x, 4x, 8x, 16x, 32x
        *   Base 2x -> Trail: 2x, 4x, 8x, 16x, 32x, 64x
        *   ...
        *   Base 32x -> Trail: 32x, 64x, 128x, 256x, 512x, 1024x (Max Trail)
    *   At the start of each Free Spin, the multiplier resets to the _first value_ of the current Active Trail (i.e., the current FS Base Multiplier).
    *   It increases one step along the current Active Trail for each Avalanche triggered by a **winning cluster**.
    *   Stops increasing at the 6th step of the current trail within a single Free Spin sequence.

### 6) Wild Symbols

*   **Standard Wild (W):**
    *   Substitutes for all paying symbols (HP, LP).
    *   Can appear naturally based on the active symbol distribution.
    *   Can be spawned via the Wild Spawning feature (see 4.7).
*   **Explosivo Wild (EW):**
    *   Substitutes for all paying symbols (HP, LP).
    *   Can appear naturally based on the active symbol distribution.
    *   Can be spawned via the Wild Spawning feature (see 4.7).
    *   **Explosion Feature:**
        *   **Trigger:** Automatically occurs _after_ win calculation if an EW is present on the grid. Both landed and spawned EWs will explode before the next avalanche refill.
        *   **Action:** The EW explodes and is removed.
        *   **Area:** Affects a 3x3 grid area centered on the EW.
        *   **Effect:** Destroys all symbols in the 3x3 area **EXCEPT** HP symbols, other Wilds (W or EW), and Scatters (S). Only LP symbols are destroyed.
        *   **Outcome:** Triggers an Avalanche. Does _not_ award a direct win for destroyed symbols. Does _not_ increment the Mucho Multiplier.
        *   **FS Collection:** If this occurs during Free Spins, the EW Collection Counter is incremented (see 4.9).

### 7) Wild Spawning

*   **Trigger:** Occurs after a winning cluster is paid and its symbols are designated for removal, but before the next Avalanche refill.
*   **Mechanic:** For _each_ winning cluster identified:
    1. A random empty position within that cluster's original footprint is selected.
    2. A Wild symbol (**W** or **EW**) is **guaranteed** to be placed in that position.
    3. The choice between W and EW is determined by probabilities `P(SpawnW)` and `P(SpawnEW)` (where `P(SpawnW) + P(SpawnEW) = 1`). _These probabilities require tuning._
    4. **Collision Handling:** If multiple clusters spawn Wilds simultaneously, and a chosen spawn location is already occupied by another Wild spawned in the _same_ event, the system must select a different available empty location from the cluster's footprint for the subsequent spawn.

### 8) Scatter Symbol & Free Spins Trigger

*   **Symbol:** Silver Robot Skull (S).
*   **Appearance:** Lands naturally based on the active symbol distribution (`P_BG(S)` or `P_FS(S)`).
*   **Base Game Trigger:** Landing 3 or more Scatter symbols anywhere on the grid during a complete spin sequence (initial drop + all Avalanches) triggers Free Spins:
    *   3 Scatters: **10 Free Spins**
    *   4 Scatters: **12 Free Spins**
    *   5+ Scatters: 12 + (N-4) * 2 Free Spins (N = number of Scatters).
*   **Free Spins Retrigger:** Landing 2 or more Scatter symbols during a Free Spin sequence awards additional spins:
    *   2 Scatters: **+3 Free Spins**
    *   3 Scatters: **+5 Free Spins**
    *   4 Scatters: **+7 Free Spins**
    *   5+ Scatters: +7 + (N-4) * 2 Free Spins.

### 9) Free Spins Feature

*   Uses the specific **FS Symbol Distribution** table.
*   Features the **Enhanced Mucho Multiplier** (see 4.5).
*   **EW Collection & Upgrade:**
    *   An **EW Collection Counter** tracks collected EWs throughout the Free Spins session.
    *   The counter increments by 1 **immediately** every time an EW symbol is removed from the grid (either by its own explosion or by being part of a winning cluster).
    *   **Upgrade Check:** At the _end_ of each completed Free Spin sequence (after all Avalanches resolve), the system checks the total number of EWs collected since the last check. For every **3** EWs collected, one upgrade is marked as 'pending'.
    *   **Upgrade Application:** At the _start_ of the _next_ Free Spin:
        *   For each 'pending' upgrade:
            *   The **FS Base Multiplier** increases one level (max 32x).
            *   **+1 additional Free Spin** is awarded.
        *   The 'pending' upgrades are then cleared.
    *   The Active Multiplier Trail for the spin is determined by the potentially updated FS Base Multiplier.
*   Scatter retriggers (see 4.8) are possible and awarded instantly.

### 10) Bet+ Options (Base Game Only)

*   Optional side bets that modify the Base Game spin cost and chances.
*   **Bonus Boost:**
    *   Cost: **1.5x** Base Bet.
    *   Effect: Approximately **doubles** the chance of triggering Free Spins (implemented by increasing `P_BG(S)`).
*   **The Enrico Show:**
    *   Cost: **2x** Base Bet.
    *   Effect: Guarantees at least **1 Explosivo Wild (EW)** lands on the _initial_ grid drop.
*   **Bonus Boost Plus:**
    *   Cost: **3x** Base Bet.
    *   Effect: Approximately **5x** the chance of triggering Free Spins (implemented by increasing `P_BG(S)` more significantly).

### 11) Feature Buy

*   Allows direct purchase of Free Spins entry from the Base Game.
*   Cost: **75x** the current Base Bet.
*   Outcome: Immediately triggers the Free Spins feature, starting with **10 spins**.
*   Target RTP for this feature specifically: **94.40%**.