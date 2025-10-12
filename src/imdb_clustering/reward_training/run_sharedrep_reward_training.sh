#!/usr/bin/env bash
set -euo pipefail

# -------- Config you can tweak --------
CONFIG="/home/mb1921/paper_repo/sharedrep-rlhf/accelerate_config.yaml"
SCRIPT="/home/mb1921/paper_repo/sharedrep-rlhf/src/imdb_clustering/reward_training/sharedrep_reward_training.py"

USER_ID="marcellobullo"
#SEED=42
EM_ITERS=10
NUM_USERS=30

# If you call the script with no args, these are the default k's:
K_VALUES=(2 4 8 16 32)
SEEDS=(42 14 28 73 100 2025 7 1234 5678 91011)

# --------------------------------------

echo "Running with: USER_ID=$USER_ID EM_ITERS=$EM_ITERS NUM_USERS=$NUM_USERS"
echo "Ks: ${K_VALUES[*]}"
echo

for SEED in "${SEEDS[@]}"; do
  echo "Setting SEED=$SEED"
    for k in "${K_VALUES[@]}"; do

    echo "======================"
    echo "Starting run: k=${k}"
    echo "======================"

    accelerate launch \
        --config_file "$CONFIG" \
        "$SCRIPT" \
        --user_id "$USER_ID" \
        --seed "$SEED" \
        --k "$k" \
        --em_iters "$EM_ITERS" \
        --num_users "$NUM_USERS"

    echo "Finished k=${k}"
    echo
    done
done
echo "All runs done."