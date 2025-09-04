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

# Gold 
echo "========================="
echo "          GOLD           "
echo "========================="
echo ""
GOLD_GRPO_SCRIPT="$PROJECT_DIR/src/helpful_harmless/grpo_training/gold_grpo_training.py"

for SEED in "${SEEDS[@]}"; do
    echo "Running GOLD GRPO training for seed $SEED"
    accelerate launch --config_file "$CONFIG_FILE" "$GOLD_GRPO_SCRIPT" --seed "$SEED" --user_id "$HF_USER_ID"
done

# Maxmin
echo "========================="
echo "         MAXMIN          "
echo "========================="
echo ""
MM_GRPO_SCRIPT="$PROJECT_DIR/src/helpful_harmless/grpo_training/maxmin_grpo_training.py"

for SEED in "${SEEDS[@]}"; do
    for PROPORTION in "${PROPORTIONS[@]}"; do
        echo "Running MAXMIN GRPO training for seed $SEED, proportion $PROPORTION"
        accelerate launch --config_file "$CONFIG_FILE" "$MM_GRPO_SCRIPT" --seed "$SEED" --proportion "$PROPORTION" --user_id "$HF_USER_ID"
    done
done

# SharedRep
echo "========================="
echo "        SHAREDREP        "
echo "========================="
echo ""
SR_GRPO_SCRIPT="$PROJECT_DIR/src/helpful_harmless/grpo_training/sharedrep_grpo_training.py"
Ks=(16)

for SEED in "${SEEDS[@]}"; do
    for PROPORTION in "${PROPORTIONS[@]}"; do
        for K in "${Ks[@]}"; do
            echo "Running SHAREDREP GRPO training for seed $SEED, proportion $PROPORTION, and K $K"
            accelerate launch --config_file "$CONFIG_FILE" "$SR_GRPO_SCRIPT" --seed "$SEED" --proportion "$PROPORTION" --k "$K" --user_id "$HF_USER_ID"
        done
    done
done