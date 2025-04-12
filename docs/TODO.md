Okay, here is a step-by-step to-do list in Markdown format for building the "Esqueleto Explosivo 3 Clone" slot game based on the PRD (derived from Math Spec Rev 4).

This list assumes a typical game development workflow, separating math/simulation from client/asset development where appropriate, though many steps will have overlap and iteration.

# To-Do List: Building "Esqueleto Explosivo 3 Clone"

This list outlines the key phases and tasks required to develop the slot game based on the provided PRD.

## Phase 1: Pre-Production & Setup

1.  **[DONE] Define Core Requirements:** Finalize the PRD based on Math Spec Rev 4.
2.  **Assemble Team:** Assign roles (Project Manager, Mathematician, Server Dev, Client Dev, Artist, Animator, Sound Designer, QA).
3.  **Choose Technology Stack:** Confirm client framework (e.g., Phaser, PixiJS, PlayCanvas), server language/platform (if applicable, e.g., Node.js, C#), simulation language (e.g., C++, Python, Java).
4.  **Setup Project Management:** Use tools like Jira, Trello, Asana to track tasks.
5.  **Setup Version Control:** Initialize Git repositories (Simulator, Server, Client).
6.  **Establish Communication Channels:** Slack, Teams, etc.

## Phase 2: Mathematical Simulation & Tuning (Critical Path)

*Goal: Validate the math model against RTP, Volatility, and Max Win targets.*

7.  **Design Simulator Architecture:** Plan how the simulator will handle state, events, and statistics gathering.
8.  **Implement Core Data Structures:** Grid representation, Symbol objects/data.
9.  **Implement Symbol Distribution:** Code functions to populate the grid based on `P_BG` and `P_FS` probability tables. Make tables easily configurable.
10. **Implement Cluster Detection Logic:** Efficiently find clusters of 5+ symbols.
11. **Implement Paytable Logic:** Function to return payout based on symbol and cluster size (configurable paytable).
12. **Implement Avalanche Logic:**
    *   Symbol removal (winning clusters, EW explosions).
    *   Wild Spawning Logic (incl. W/EW probability, collision handling).
    *   Symbol dropping/falling.
    *   Grid refilling using active distribution.
13. **Implement Mucho Multiplier Logic:**
    *   Base Game sequence (1x-32x).
    *   Free Spins sequence (based on FS Base Multiplier, 6 steps, max 1024x).
    *   Correct increment trigger (on win-based avalanche).
    *   Correct reset logic (new spin / new Free Spin).
14. **Implement Explosivo Wild (EW) Logic:**
    *   Substitution.
    *   Explosion trigger (on landing).
    *   Explosion area effect (destroying correct symbols).
15. **Implement Scatter Logic & Free Spins Trigger:** Detect Scatter counts over a full spin sequence (incl. avalanches) to trigger FS.
16. **Implement Free Spins Feature Logic:** ✅ Done (Spin count, EW collection, Upgrades, Retriggers handled in `run_free_spins_feature`).
17. **Implement Bet+ Option Logic:** ⏳ **POSTPONED.** Simulate modifications (P(S) increase for Bonus Boosts, guaranteed EW for Enrico Show).
18. **Implement Feature Buy Logic:** ⏳ **POSTPONED.** Simulate direct entry into FS (10 spins).
19. **Implement Statistics Tracking:** ✅ Done (RTP, hit freq, FS freq, avg FS win, win distribution, etc., are tracked and logged in `run_simulation`).
20. **Initial Parameterization:** Input starting paytable values, estimated BG/FS probabilities, and Wild Spawn probabilities from the spec.
21. **Run & Analyze Initial Simulations:** Execute millions/billions of spins for BG, FS (from trigger), and Feature Buy scenarios.
22. **Tune Parameters & Iterate:**
    *   Adjust Paytable values.
    *   Adjust `P_BG` and `P_FS` (especially Scatter, EW, Wild probabilities).
    *   Adjust `P(SpawnW)` / `P(SpawnEW)` ratio.
    *   Re-run simulations (Step 21).
    *   Repeat until all target metrics (RTPs, Volatility, Max Win) are met and frequencies feel appropriate.
23. **[Milestone] Math Model Validation:** Lock down the final, validated paytable and probability parameters. Document these carefully.

## Phase 3: Asset Production (Parallel Task)

*Goal: Create all visual and audio assets based on the theme and specs.*

24. **Define Art Style Guide:** Establish the visual direction (color palette, character style, mood).
25. **Create Concept Art:** Sketches for background, symbols, characters, UI layout.
26. **Produce Final Background Art:** Main game background, Free Spins variant (if any).
27. **Produce Final Symbol Art:** HP, LP symbols, Wild (W), Explosivo Wild (EW), Scatter (S). Ensure clarity and distinction.
28. **Produce UI Elements:** Buttons (Spin, Bet+/-, Buy, Info, Settings), Frames, Meters (Multiplier, EW Collection), Win Displays, Balance/Bet/Win text areas.
29. **Create Animations:**
    *   Symbol landing/idle animations (subtle).
    *   Avalanche/symbol drop animations.
    *   Cluster win highlights/animations.
    *   Wild spawn animations.
    *   EW explosion visual effect.
    *   Multiplier meter updates/animations.
    *   FS Trigger transition animation.
    *   Character animations (ambient, reactive).
30. **Create Static Screens:** Loading screen, Paytable/Info screens.
31. **Compose Music:** Background loops for Base Game and Free Spins (distinct moods).
32. **Create/Source Sound Effects:** Spin start/stop, symbol landings, cluster wins (various sizes), avalanche whooshes, Wild spawn sounds, EW explosion sound, multiplier increments, Scatter landings, FS trigger fanfare, UI button clicks.
33. **Master Audio:** Ensure consistent levels and quality across all audio assets.

## Phase 4: Game Implementation (Client & Server)

*Goal: Build the playable game using the validated math and created assets.*

34. **Setup Client Project:** Initialize project using chosen framework (Phaser, PixiJS, etc.).
35. **Implement Server Logic (If Applicable):**
    *   Set up secure server environment.
    *   Implement **certified** RNG.
    *   Port the **exact** validated math logic from the simulator to the server.
    *   Implement game state management (balance, FS state, counters).
    *   Develop secure API for client communication (spin requests, results).
    *   Implement session handling.
36. **Implement Client Framework:**
    *   Basic game states/scenes (Loading, BaseGame, FreeSpins, Info).
    *   Asset loading system.
37. **Implement Grid & Symbol Rendering:** Display the 5x5 grid and symbols based on data received.
38. **Implement Communication Layer:** Client-side logic to send requests (spin, buy) and handle responses from the server/math engine.
39. **Implement Core Gameplay Loop (Client):**
    *   Trigger spin animation.
    *   Display initial symbol drop.
    *   Animate cluster wins based on results.
    *   Animate Wild spawns based on results.
    *   Animate EW explosions based on results.
    *   Animate Avalanches (symbol removal, falling, refilling).
    *   Handle the entire sequence based on results data.
40. **Implement UI:**
    *   Display Balance, Bet, Win amounts.
    *   Implement bet adjustment controls.
    *   Implement Spin button functionality.
    *   Implement Bet+ option buttons and state indicators.
    *   Implement Feature Buy button and confirmation flow.
    *   Display Mucho Multiplier value/meter and updates.
    *   Display Free Spins count.
    *   Display EW Collection meter/progress during FS.
    *   Implement Info/Paytable screen display.
41. **Integrate Assets:**
    *   Map final art assets to symbols, background, UI.
    *   Integrate animations triggered by game events.
    *   Integrate sound effects and music playback triggered by game events.
42. **Ensure Responsive Design:** Adapt layout for different screen sizes and orientations (desktop/mobile).

## Phase 5: Testing & Quality Assurance

*Goal: Ensure the game is bug-free, performs well, meets requirements, and is enjoyable.*

43. **Develop Test Plan:** Outline test cases for functionality, math verification, UI/UX, performance, compatibility.
44. **Functional Testing:**
    *   Verify all game mechanics work as per the PRD on client/server.
    *   Test cluster formation and payouts.
    *   Test Avalanche sequences exhaustively.
    *   Test Wild substitution and spawning.
    *   Test EW explosion trigger and effect.
    *   Test Multiplier progression and application (BG & FS).
    *   Test FS trigger, EW collection, Base Multiplier upgrade, spin awards.
    *   Test Scatter retriggers.
    *   Test Bet+ options functionality and cost.
    *   Test Feature Buy entry and cost.
    *   Test edge cases (e.g., multiple simultaneous clusters/spawns, max win).
45. **Math Verification (Client/Server vs Simulator):** Run automated tests or specific scenarios on the final game build to ensure results *exactly* match the validated simulator output for identical inputs/RNG seeds.
46. **UI/UX Testing:** Check layout, readability, button responsiveness, clarity of information (wins, multipliers, FS progress).
47. **Visual & Audio Testing:** Verify all animations play correctly, visual effects match events, audio cues are timed correctly and appropriate.
48. **Compatibility Testing:** Test on target browsers (Chrome, Firefox, Safari, Edge) and devices (iOS, Android - various models/OS versions).
49. **Performance Testing:** Monitor frame rate (FPS) and loading times, especially during complex avalanche sequences. Optimize as needed.
50. **Regression Testing:** Re-test previously fixed bugs after new changes are implemented.
51. **Compliance Testing (If Real Money):** Submit game/RNG for certification by relevant testing labs if required for target jurisdictions.
52. **Bug Tracking & Resolution:** Log all bugs found, prioritize, fix, and verify fixes.

## Phase 6: Deployment & Release

*Goal: Launch the game to the target audience.*

53. **Final Build Preparation:** Code freeze, build optimization, package creation.
54. **Staging Deployment:** Deploy the final build to a pre-production environment.
55. **Final QA / Smoke Test:** Perform final checks on the staging environment.
56. **Production Deployment:** Release the game to live servers / platform(s).
57. **Marketing & Announcement:** Coordinate launch communication.

## Phase 7: Post-Release Monitoring

*Goal: Ensure stability and gather feedback after launch.*

58. **Monitor Performance & Errors:** Track server load, game errors, and performance metrics in the live environment.
59. **Gather Player Feedback:** Monitor forums, support channels, reviews.
60. **Plan Updates:** Schedule potential bug fix patches or minor improvements based on monitoring and feedback.

---
**Note:** This is a sequential list, but many phases (especially Asset Production, Implementation, and Testing) will have significant overlap and require continuous communication and iteration between teams. The Math Simulation phase is foundational and should reach validation before locking down core game logic implementation.
