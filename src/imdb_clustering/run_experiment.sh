#!/usr/bin/env bash
set -euo pipefail

# Get the directory where this bash script lives
SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_DIR="$(realpath "$SCRIPT_DIR/../..")"

echo "Project directory: $PROJECT_DIR"

# Sccelerate Config
CONFIG_FILE="$PROJECT_DIR/accelerate_config.yaml"

# Parameters
USER_ID="marcellobullo"
EM_ITERS=10
NUM_USERS=30

K_VALUES=(2 32)
MINPROPS=(0.05 0.2)
SEEDS=(42 14 28 73 100 2025 7 1234 5678 91011)

# EXPERIMENT
for SEED in "${SEEDS[@]}"; do
  for MINPROP in "${MINPROPS[@]}"; do

    #############################
    #      REWARD TRAINING      #
    #############################
    
    ##### SharedRep ############################################################################################################################################
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
    ############################################################################################################################################################

    ##### Maxmin ###############################################################################################################################################
    MM_RM_SCRIPT="$PROJECT_DIR/src/imdb_clustering/reward_training/maxmin_reward_training.py"
    accelerate launch --config_file "$CONFIG_FILE" "$MM_RM_SCRIPT" \
      --user_id "$USER_ID" \
      --seed "$SEED" \
      --minprop "$MINPROP" \
      --em_iters "$EM_ITERS" \
      --num_users "$NUM_USERS"
    ############################################################################################################################################################



    ##########################
    #      PPO TRAINING      #
    ##########################
    
    ##### GOLD ################################################################################################################################################
    GOLD_PPO_SCRIPT="$PROJECT_DIR/src/imdb_clustering/ppo_training/gold_ppo_training.py"
    echo "Running GOLD PPO training for seed $SEED"
    # Gold PPO Training
    accelerate launch --config_file "$CONFIG_FILE" "$GOLD_PPO_SCRIPT" --seed "$SEED" --user_id "$USER_ID"

    # Gold Evaluation
    EVAL_SCRIPT="$PROJECT_DIR/src/imdb_clustering/evaluation/generate_responses.py"
    SCORE_SCRIPT="$PROJECT_DIR/src/imdb_clustering/evaluation/score_responses.py"
    accelerate launch --config_file "$CONFIG_FILE" "$EVAL_SCRIPT" --user_id "$USER_ID" --seed "$SEED" --method "gold" --proportion "$MINPROP" --k "$K"
    accelerate launch --config_file "$CONFIG_FILE" "$SCORE_SCRIPT" --user_id "$USER_ID" --seed "$SEED" --method "gold" --proportion "$MINPROP" --k "$K"
    ###########################################################################################################################################################

    ##### SHAREDREP ###########################################################################################################################################
    SR_PPO_SCRIPT="$PROJECT_DIR/src/imdb_clustering/ppo_training/sharedrep_ppo_training.py"
    for K in "${K_VALUES[@]}"; do
      # Sharedrep PPO Training
      echo "Running SHAREDREP PPO training for seed $SEED and K $K"
      accelerate launch --config_file "$CONFIG_FILE" "$SR_PPO_SCRIPT" --seed "$SEED" --k "$K" --user_id "$USER_ID" --minprop "$MINPROP"

      # Sharedrep Evaluation
      EVAL_SCRIPT="$PROJECT_DIR/src/imdb_clustering/evaluation/generate_responses.py"
      SCORE_SCRIPT="$PROJECT_DIR/src/imdb_clustering/evaluation/score_responses.py"
      accelerate launch --config_file "$CONFIG_FILE" "$EVAL_SCRIPT" --user_id "$USER_ID" --seed "$SEED" --method "sharedrep" --proportion "$MINPROP" --k "$K"
      accelerate launch --config_file "$CONFIG_FILE" "$SCORE_SCRIPT" --user_id "$USER_ID" --seed "$SEED" --method "sharedrep" --proportion "$MINPROP" --k "$K"
    done
    ###########################################################################################################################################################

    ##### MAXMIN ##############################################################################################################################################
    # Maxmin PPO Training
    MM_PPO_SCRIPT="$PROJECT_DIR/src/imdb_clustering/ppo_training/maxmin_ppo_training.py"
    echo "Running MAXMIN PPO training for seed $SEED"
    accelerate launch --config_file "$CONFIG_FILE" "$MM_PPO_SCRIPT" --seed "$SEED" --user_id "$USER_ID" --minprop "$MINPROP"

    # Maxmin Evaluation
    EVAL_SCRIPT="$PROJECT_DIR/src/imdb_clustering/evaluation/generate_responses.py"
    SCORE_SCRIPT="$PROJECT_DIR/src/imdb_clustering/evaluation/score_responses.py"
    accelerate launch --config_file "$CONFIG_FILE" "$EVAL_SCRIPT" --user_id "$USER_ID" --seed "$SEED" --method "maxmin" --proportion "$MINPROP" --k -1
    accelerate launch --config_file "$CONFIG_FILE" "$SCORE_SCRIPT" --user_id "$USER_ID" --seed "$SEED" --method "maxmin" --proportion "$MINPROP" --k -1
    ###########################################################################################################################################################
  done
done
echo "All runs done."