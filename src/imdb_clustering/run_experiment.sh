#!/usr/bin/env bash
set -euo pipefail

# Get the directory where this bash script lives
SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_DIR="$(realpath "$SCRIPT_DIR/../..")"

echo "Project directory: $PROJECT_DIR"

CONFIG_FILE="$PROJECT_DIR/accelerate_config.yaml"

USER_ID="marcellobullo"
EM_ITERS=10
NUM_USERS=30

#K_VALUES=(2 4 8 16 32)
K_VALUES=(2 32)
MINPROPS=(0.05 0.2)
SEEDS=(42 14 28 73 100 2025 7 1234 5678 91011)

# --------------------------------------

echo "Running with: USER_ID=$USER_ID EM_ITERS=$EM_ITERS NUM_USERS=$NUM_USERS"
echo

for SEED in "${SEEDS[@]}"; do
  for MINPROP in "${MINPROPS[@]}"; do

    #############################
    #      REWARD TRAINING      #
    #############################
    
    # SharedRep
    echo "========================="
    echo "        SHAREDREP        "
    echo "========================="
    echo ""
    echo "Setting SEED=$SEED"
    SR_RM_SCRIPT="$PROJECT_DIR/src/imdb_clustering/reward_training/sharedrep_reward_training.py"
    for k in "${K_VALUES[@]}"; do
      accelerate launch --config_file "$CONFIG_FILE" "$SR_RM_SCRIPT" \
        --user_id "$USER_ID" \
        --seed "$SEED" \
        --k "$k" \
        --minprop "$MINPROP" \
        --em_iters "$EM_ITERS" \
        --num_users "$NUM_USERS"
    done

    # Maxmin
    echo "========================="
    echo "         MAXMIN          "
    echo "========================="
    echo ""
    MM_RM_SCRIPT="$PROJECT_DIR/src/imdb_clustering/reward_training/maxmin_reward_training.py"
    accelerate launch --config_file "$CONFIG_FILE" "$MM_RM_SCRIPT" \
      --user_id "$USER_ID" \
      --seed "$SEED" \
      --minprop "$MINPROP" \
      --em_iters "$EM_ITERS" \
      --num_users "$NUM_USERS"

    ###########################
    #       PPO TRAINING      #
    ###########################
    
    # # Gold 
    # echo "========================="
    # echo "          GOLD           "
    # echo "========================="
    # echo ""
    # GOLD_PPO_SCRIPT="$PROJECT_DIR/src/imdb_clustering/ppo_training/gold_ppo_training.py"
    # echo "Running GOLD PPO training for seed $SEED"
    # accelerate launch --config_file "$CONFIG_FILE" "$GOLD_PPO_SCRIPT" --seed "$SEED" --user_id "$USER_ID"

    # SharedRep
    echo "========================="
    echo "        SHAREDREP        "
    echo "========================="
    echo ""
    SR_PPO_SCRIPT="$PROJECT_DIR/src/imdb_clustering/ppo_training/sharedrep_ppo_training.py"
    for K in "${K_VALUES[@]}"; do
        echo "Running SHAREDREP PPO training for seed $SEED and K $K"
        accelerate launch --config_file "$CONFIG_FILE" "$SR_PPO_SCRIPT" --seed "$SEED" --k "$K" --user_id "$USER_ID" --minprop "$MINPROP"
    done

    # Maxmin
    echo "========================="
    echo "         MAXMIN          "
    echo "========================="
    echo ""
    MM_PPO_SCRIPT="$PROJECT_DIR/src/imdb_clustering/ppo_training/maxmin_ppo_training.py"
    echo "Running MAXMIN PPO training for seed $SEED"
    accelerate launch --config_file "$CONFIG_FILE" "$MM_PPO_SCRIPT" --seed "$SEED" --user_id "$USER_ID" --minprop "$MINPROP"

done
echo "All runs done."