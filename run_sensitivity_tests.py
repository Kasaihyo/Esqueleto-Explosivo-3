import subprocess
import re
import os
import copy
import math

# --- Configuration ---
CONFIG_FILE_PATH = "simulator/config.py"
# Adjust command if needed. Using 100k spins for reasonable speed.
SIMULATION_COMMAND = "./run.sh 100000 --roe --roe-use-main-data"
# Updated Regex Patterns
REGEX_PATTERNS = {
    "RTP": r"Return to Player \(RTP\):\s*([0-9.]+)%",
    "HitRate": r"Hit Frequency:\s*([0-9.]+)%",
    "ROE_Median": r"Median ROE:\s*(\d+|Infinite)",
    "ROE_Average": r"Average ROE:\s*(\d+|Infinite)",
    "MaxWin": r"Max Win:.*?\(\s*([0-9,.]+)\s*x\)",
}

# --- Baseline Weights ---
BASELINE_BG = {
    "LADY_SK": 100, "PINK_SK": 250, "GREEN_SK": 250, "BLUE_SK": 300,
    "ORANGE_SK": 356, "CYAN_SK": 400, "WILD": 32, "E_WILD": 32, "SCATTER": 15,
}
BASELINE_FS = {
    "LADY_SK": 100, "PINK_SK": 250, "GREEN_SK": 250, "BLUE_SK": 300,
    "ORANGE_SK": 356, "CYAN_SK": 400, "WILD": 32, "E_WILD": 32, "SCATTER": 15,
}

# --- Target Test Configuration ---

target_bg = copy.deepcopy(BASELINE_BG)
target_fs = copy.deepcopy(BASELINE_FS)

# 1. Slightly Reduce LP Weights (-5% BG & FS)
lp_keys = ["PINK_SK", "GREEN_SK", "BLUE_SK", "ORANGE_SK", "CYAN_SK"]
for key in lp_keys:
    target_bg[key] = max(1, math.ceil(target_bg[key] * 0.95))
    target_fs[key] = max(1, math.ceil(target_fs[key] * 0.95))

# 2. Slightly Increase FS Wild Weights (+10% FS)
target_fs["WILD"] = math.ceil(target_fs["WILD"] * 1.10)
target_fs["E_WILD"] = math.ceil(target_fs["E_WILD"] * 1.10)

# --- Test Configurations to Run ---
# Modified to only run Baseline and the target test
TEST_CONFIGURATIONS = {
    "0_Baseline": (copy.deepcopy(BASELINE_BG), copy.deepcopy(BASELINE_FS)),
    "Target_RTP95_ROE700": (target_bg, target_fs),
}

# Function to modify weights, ensuring they are integers
def modify_weights(weights, modifications):
    new_weights = copy.deepcopy(weights)
    for symbol, change_percent in modifications.items():
        if symbol in new_weights:
            original = new_weights[symbol]
            modified = max(1, math.ceil(original * (1 + change_percent / 100.0))) # Ensure weight >= 1
            new_weights[symbol] = modified
    return new_weights

# Config 1 (BG HP+15%)
bg1, fs1 = copy.deepcopy(BASELINE_BG), copy.deepcopy(BASELINE_FS)
bg1["LADY_SK"] = math.ceil(bg1["LADY_SK"] * 1.15)
TEST_CONFIGURATIONS["1_BG_HP+15%"] = (bg1, fs1)

# Config 2 (BG LP-10%)
bg2, fs2 = copy.deepcopy(BASELINE_BG), copy.deepcopy(BASELINE_FS)
lp_keys = ["PINK_SK", "GREEN_SK", "BLUE_SK", "ORANGE_SK", "CYAN_SK"]
for key in lp_keys: bg2[key] = max(1, math.ceil(bg2[key] * 0.90))
TEST_CONFIGURATIONS["2_BG_LP-10%"] = (bg2, fs2)

# Config 3 (BG Wilds+30%)
bg3, fs3 = copy.deepcopy(BASELINE_BG), copy.deepcopy(BASELINE_FS)
bg3["WILD"] = math.ceil(bg3["WILD"] * 1.30)
bg3["E_WILD"] = math.ceil(bg3["E_WILD"] * 1.30)
TEST_CONFIGURATIONS["3_BG_Wilds+30%"] = (bg3, fs3)

# Config 4 (BG Scatter+30%)
bg4, fs4 = copy.deepcopy(BASELINE_BG), copy.deepcopy(BASELINE_FS)
bg4["SCATTER"] = math.ceil(bg4["SCATTER"] * 1.30)
TEST_CONFIGURATIONS["4_BG_Scatter+30%"] = (bg4, fs4)

# Config 5 (FS HP+15%)
bg5, fs5 = copy.deepcopy(BASELINE_BG), copy.deepcopy(BASELINE_FS)
fs5["LADY_SK"] = math.ceil(fs5["LADY_SK"] * 1.15)
TEST_CONFIGURATIONS["5_FS_HP+15%"] = (bg5, fs5)

# Config 6 (FS LP-10%)
bg6, fs6 = copy.deepcopy(BASELINE_BG), copy.deepcopy(BASELINE_FS)
for key in lp_keys: fs6[key] = max(1, math.ceil(fs6[key] * 0.90))
TEST_CONFIGURATIONS["6_FS_LP-10%"] = (bg6, fs6)

# Config 7 (FS Wilds+30%)
bg7, fs7 = copy.deepcopy(BASELINE_BG), copy.deepcopy(BASELINE_FS)
fs7["WILD"] = math.ceil(fs7["WILD"] * 1.30)
fs7["E_WILD"] = math.ceil(fs7["E_WILD"] * 1.30)
TEST_CONFIGURATIONS["7_FS_Wilds+30%"] = (bg7, fs7)

# Config 8 (FS Scatter+30%)
bg8, fs8 = copy.deepcopy(BASELINE_BG), copy.deepcopy(BASELINE_FS)
fs8["SCATTER"] = math.ceil(fs8["SCATTER"] * 1.30)
TEST_CONFIGURATIONS["8_FS_Scatter+30%"] = (bg8, fs8)

# Config 9 (FS Richer)
bg9, fs9 = copy.deepcopy(BASELINE_BG), copy.deepcopy(BASELINE_FS)
fs9["LADY_SK"] = math.ceil(fs9["LADY_SK"] * 1.15)
fs9["WILD"] = math.ceil(fs9["WILD"] * 1.15)
fs9["E_WILD"] = math.ceil(fs9["E_WILD"] * 1.15)
for key in lp_keys: fs9[key] = max(1, math.ceil(fs9[key] * 0.90))
TEST_CONFIGURATIONS["9_FS_Richer"] = (bg9, fs9)

# Config 10 (FS Poorer)
bg10, fs10 = copy.deepcopy(BASELINE_BG), copy.deepcopy(BASELINE_FS)
fs10["LADY_SK"] = max(1, math.ceil(fs10["LADY_SK"] * 0.85))
fs10["WILD"] = max(1, math.ceil(fs10["WILD"] * 0.85))
fs10["E_WILD"] = max(1, math.ceil(fs10["E_WILD"] * 0.85))
for key in lp_keys: fs10[key] = math.ceil(fs10[key] * 1.10)
TEST_CONFIGURATIONS["10_FS_Poorer"] = (bg10, fs10)

# Config 11 (Shift BG->FS)
bg11, fs11 = copy.deepcopy(BASELINE_BG), copy.deepcopy(BASELINE_FS)
rich_keys = ["LADY_SK", "WILD", "E_WILD"]
for key in rich_keys:
    bg11[key] = max(1, math.ceil(bg11[key] * 0.90))
    fs11[key] = math.ceil(fs11[key] * 1.10)
TEST_CONFIGURATIONS["11_Shift_BG->FS"] = (bg11, fs11)

# Config 12 (Shift FS->BG)
bg12, fs12 = copy.deepcopy(BASELINE_BG), copy.deepcopy(BASELINE_FS)
for key in rich_keys:
    bg12[key] = math.ceil(bg12[key] * 1.10)
    fs12[key] = max(1, math.ceil(fs12[key] * 0.90))
TEST_CONFIGURATIONS["12_Shift_FS->BG"] = (bg12, fs12)

# --- Helper Functions ---

def dict_to_str(d):
    """Converts a dictionary to a pretty-printed string representation."""
    items_str = ",\n    ".join(f'"{k}": {v}' for k, v in d.items())
    return f"{{\n    {items_str},\n}}"

def modify_config_file(config_path, bg_weights, fs_weights):
    """Reads config, replaces weight dicts, writes back."""
    try:
        with open(config_path, 'r') as f:
            content = f.read()

        # Replace BG weights
        content = re.sub(
            r"SYMBOL_GENERATION_WEIGHTS_BG\s*=\s*\{.*?\n\}",
            f"SYMBOL_GENERATION_WEIGHTS_BG = {dict_to_str(bg_weights)}",
            content,
            flags=re.DOTALL
        )
        # Replace FS weights
        content = re.sub(
            r"SYMBOL_GENERATION_WEIGHTS_FS\s*=\s*\{.*?\n\}",
            f"SYMBOL_GENERATION_WEIGHTS_FS = {dict_to_str(fs_weights)}",
            content,
            flags=re.DOTALL
        )

        with open(config_path, 'w') as f:
            f.write(content)
        return True
    except Exception as e:
        print(f"Error modifying config file: {e}")
        return False

def run_simulation():
    """Runs the simulation command and returns output."""
    try:
        print(f"Running command: {SIMULATION_COMMAND}")
        result = subprocess.run(
            SIMULATION_COMMAND,
            shell=True,
            capture_output=True,
            text=True,
            check=True,
            cwd=os.path.dirname(os.path.abspath(__file__)) # Run from script's dir
        )
        print("Simulation finished.")
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Simulation command failed with error code {e.returncode}")
        print(f"Stderr: {e.stderr}")
        print(f"Stdout: {e.stdout}")
        return None
    except Exception as e:
        print(f"Error running simulation: {e}")
        return None

def parse_results(output_text):
    """Parses simulation output using regex."""
    results = {}
    if not output_text:
        return {k: "Error" for k in REGEX_PATTERNS}

    for key, pattern in REGEX_PATTERNS.items():
        match = re.search(pattern, output_text)
        results[key] = match.group(1) if match else "N/A"

    for key in REGEX_PATTERNS:
        if key not in results:
            results[key] = "N/A"

    return results

# --- Main Execution ---
if __name__ == "__main__":
    original_config_content = None
    results_data = []

    try:
        # 1. Backup original config
        print(f"Reading original config from {CONFIG_FILE_PATH}")
        with open(CONFIG_FILE_PATH, 'r') as f:
            original_config_content = f.read()
        print("Original config backed up.")

        # 2. Run tests (Now only Baseline and Target)
        for name, (bg_weights, fs_weights) in TEST_CONFIGURATIONS.items():
            print(f"\n--- Running Test: {name} ---")

            # Modify config
            if not modify_config_file(CONFIG_FILE_PATH, bg_weights, fs_weights):
                print("Skipping simulation due to config modification error.")
                error_result = {k: "Error" for k in REGEX_PATTERNS}
                error_result["Config"] = name
                results_data.append(error_result)
                continue

            print("Config file modified.")

            # Run simulation
            output = run_simulation()

            # Parse results
            if output:
                parsed = parse_results(output)
                parsed["Config"] = name
                results_data.append(parsed)
                print(f"Results: {parsed}")
            else:
                print("Simulation run failed or produced no output.")
                failed_result = {k: "Failed" for k in REGEX_PATTERNS}
                failed_result["Config"] = name
                results_data.append(failed_result)

    except Exception as e:
        print(f"\nAn error occurred during the test execution: {e}")
    finally:
        # 3. Restore original config
        if original_config_content:
            print(f"\nRestoring original content to {CONFIG_FILE_PATH}")
            try:
                with open(CONFIG_FILE_PATH, 'w') as f:
                    f.write(original_config_content)
                print("Original config restored successfully.")
            except Exception as e:
                print(f"FATAL: Failed to restore original config: {e}")
                print("You may need to restore it manually from backup or version control.")
        else:
            print("\nWarning: Could not restore config file (backup was not created).")

    # 4. Print summary table
    if results_data:
        print("\n--- Targeted Test Results Summary ---")
        header = ["Config", "RTP (%)", "HitRate (%)", "ROE Median", "ROE Average", "MaxWin (x)"]
        print(f"{header[0]:<22} {header[1]:>10} {header[2]:>12} {header[3]:>11} {header[4]:>11} {header[5]:>12}")
        print("-" * 85) # Adjusted width for longer config name
        for result in results_data:
            print(f"{result.get('Config', 'N/A'):<22} "
                  f"{result.get('RTP', 'N/A'):>10} "
                  f"{result.get('HitRate', 'N/A'):>12} "
                  f"{result.get('ROE_Median', 'N/A'):>11} "
                  f"{result.get('ROE_Average', 'N/A'):>11} "
                  f"{result.get('MaxWin', 'N/A'):>12}")
        print("-" * 85) # Adjusted width

    print("\nTargeted analysis complete.") 