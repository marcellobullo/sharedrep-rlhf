#!/bin/bash

# Get the directory where this bash script lives
SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_DIR="$(realpath "$SCRIPT_DIR/../../..")"

# Accelerate configuration path
CONFIG_FILE="$PROJECT_DIR/accelerate_config.yaml"

# Parameters
HF_USER_ID="marcellobullo"
SEEDS=(42)
PROPORTIONS=(0.1)

# SharedRep
echo "========================="
echo "        SHAREDREP        "
echo "========================="
echo ""
SR_SCRIPT="$PROJECT_DIR/src/helpful_harmless/reward_training/sharedrep_reward_training.py"
Ks=(2 16)

for SEED in "${SEEDS[@]}"; do
    for PROPORTION in "${PROPORTIONS[@]}"; do
        for K in "${Ks[@]}"; do
            echo "Running SHAREDREP Reward training for seed $SEED, proportion $PROPORTION, and K $K"
            accelerate launch --config_file "$CONFIG_FILE" "$SR_SCRIPT" --seed "$SEED" --proportion "$PROPORTION" --k "$K" --user_id "$HF_USER_ID"
        done
    done
done

# Maxmin 
MM_SCRIPT="$PROJECT_DIR/src/helpful_harmless/reward_training/maxmin_reward_training.py"
GIDS=(0 1)
echo "========================="
echo "         MAXMIN          "
echo "========================="
echo ""

for SEED in "${SEEDS[@]}"; do
    for PROPORTION in "${PROPORTIONS[@]}"; do
        for GID in "${GIDS[@]}"; do
            echo "Running MAXMIN Reward training for seed $SEED, proportion $PROPORTION"
            accelerate launch --config_file "$CONFIG_FILE" "$MM_SCRIPT" --seed "$SEED" --proportion "$PROPORTION" --group "$GID" --user_id "$HF_USER_ID"
        done
    done
done



        

