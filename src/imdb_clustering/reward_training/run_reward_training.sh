#!/usr/bin/env bash
set -euo pipefail

# Get the directory where this bash script lives
SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_DIR="$(realpath "$SCRIPT_DIR/../../..")"

CONFIG="$PROJECT_DIR/accelerate_config.yaml"
SCRIPT="$PROJECT_DIR/src/imdb_clustering/reward_training/maxmin_reward_training.py"

USER_ID="marcellobullo"
EM_ITERS=10
NUM_USERS=30

K_VALUES=(2 4 8 16 32)
SEEDS=(42 14 28 73 100 2025 7 1234 5678 91011)

# --------------------------------------

echo "Running with: USER_ID=$USER_ID EM_ITERS=$EM_ITERS NUM_USERS=$NUM_USERS"
echo

for SEED in "${SEEDS[@]}"; do
  
  # SharedRep
  echo "========================="
  echo "        SHAREDREP        "
  echo "========================="
  echo ""
  echo "Setting SEED=$SEED"
  SR_RM_SCRIPT="$PROJECT_DIR/src/imdb_clustering/reward_training/sharedrep_reward_training.py"
  for k in "${K_VALUES[@]}"; do
    accelerate launch --config_file "$CONFIG" "$SR_RM_SCRIPT" \
      --user_id "$USER_ID" \
      --seed "$SEED" \
      --k "$k" \
      --em_iters "$EM_ITERS" \
      --num_users "$NUM_USERS"
  done

  # Maxmin
  echo "========================="
  echo "         MAXMIN          "
  echo "========================="
  echo ""
  MM_RM_SCRIPT="$PROJECT_DIR/src/imdb_clustering/reward_training/maxmin_reward_training.py"
  accelerate launch --config_file "$CONFIG" "$MM_RM_SCRIPT" \
    --user_id "$USER_ID" \
    --seed "$SEED" \
    --em_iters "$EM_ITERS" \
    --num_users "$NUM_USERS"

done
echo "All runs done."