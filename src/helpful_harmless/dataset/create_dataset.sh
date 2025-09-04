#!/bin/bash

# Get the directory where this bash script lives
SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_DIR="$(realpath "$SCRIPT_DIR/../../..")"

# Script paths
GEN_SCRIPT="$PROJECT_DIR/src/helpful_harmless/dataset/generate_responses.py"
SCORE_SCRIPT="$PROJECT_DIR/src/helpful_harmless/dataset/score_responses.py"
CHOSEN_REJECTED_SCRIPT="$PROJECT_DIR/src/helpful_harmless/dataset/create_chosen_rejected.py"

# Accelerate configuration path
CONFIG_FILE="$PROJECT_DIR/accelerate_config.yaml"

# Hugging Face user ID
HF_USER_ID="YOUR_HF_USER_ID"

echo "========================="
echo "      CREATE DATASET     "
echo "========================="
echo ""

# Generate responses and score them separately to avoid memory issues
accelerate launch --config_file "$CONFIG_FILE" "$GEN_SCRIPT" --user_id "$HF_USER_ID"
accelerate launch --config_file "$CONFIG_FILE" "$SCORE_SCRIPT" --user_id "$HF_USER_ID"

# Create chosen-rejected datasets for different seeds and proportions
SEEDS=(42)
PROPORTIONS=(0.01 0.05 0.1 0.15 0.2)

for SEED in "${SEEDS[@]}"; do
  for PROPORTION in "${PROPORTIONS[@]}"; do
    python "$CHOSEN_REJECTED_SCRIPT" --user_id "$HF_USER_ID" --seed "$SEED" --proportion "$PROPORTION"
  done
done
