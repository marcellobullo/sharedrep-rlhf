#!/usr/bin/env bash
set -euo pipefail

# Get the directory where this bash script lives
SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_DIR="$(realpath "$SCRIPT_DIR/../../..")"

echo "Project directory: $PROJECT_DIR"

# Config and script paths
CONFIG_FILE="$PROJECT_DIR/accelerate_config.yaml"

# Parameters
HF_USER_ID="marcellobullo"
SEEDS=(42 14 28 73 100 2025 7 1234 5678 91011)


for SEED in "${SEEDS[@]}"; do
    
    # Gold 
    echo "========================="
    echo "          GOLD           "
    echo "========================="
    echo ""
    GOLD_PPO_SCRIPT="$PROJECT_DIR/src/imdb_clustering/ppo_training/gold_ppo_training.py"
    echo "Running GOLD PPO training for seed $SEED"
    accelerate launch --config_file "$CONFIG_FILE" "$GOLD_PPO_SCRIPT" --seed "$SEED" --user_id "$HF_USER_ID"

    # SharedRep
    echo "========================="
    echo "        SHAREDREP        "
    echo "========================="
    echo ""
    SR_PPO_SCRIPT="$PROJECT_DIR/src/imdb_clustering/ppo_training/sharedrep_ppo_training.py"
    Ks=(2 4 8 16 32)
    for K in "${Ks[@]}"; do
        echo "Running SHAREDREP PPO training for seed $SEED and K $K"
        accelerate launch --config_file "$CONFIG_FILE" "$SR_PPO_SCRIPT" --seed "$SEED" --k "$K" --user_id "$HF_USER_ID"
    done

    # Maxmin
    echo "========================="
    echo "         MAXMIN          "
    echo "========================="
    echo ""
    MM_PPO_SCRIPT="$PROJECT_DIR/src/imdb_clustering/ppo_training/maxmin_ppo_training.py"
    echo "Running MAXMIN PPO training for seed $SEED"
    accelerate launch --config_file "$CONFIG_FILE" "$MM_PPO_SCRIPT" --seed "$SEED" --user_id "$HF_USER_ID"
done