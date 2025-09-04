#!/bin/bash

# Get the directory where this bash script lives
SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_DIR="$(realpath "$SCRIPT_DIR/../../..")"
SCRIPT="$PROJECT_DIR/src/helpful_harmless/sft/sft_tinyllama_hh.py"
CONFIG_FILE="$PROJECT_DIR/accelerate_config.yaml"

HF_USER_ID="YOUR_HF_USER_ID"

echo "========================="
echo "       SFT TRAINING      "
echo "========================="
echo ""

accelerate launch --config_file "$CONFIG_FILE" "$SCRIPT" --user_id "$HF_USER_ID"
