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
Ks=(16)

# Scripts
GENERATION_SCRIPT="$PROJECT_DIR/src/helpful_harmless/evaluation/generate_responses.py"
SCORING_SCRIPT="$PROJECT_DIR/src/helpful_harmless/evaluation/score_responses.py"


# Gold 
echo "========================="
echo "          GOLD           "
echo "========================="
echo ""

for SEED in "${SEEDS[@]}"; do
    echo "Running GOLD evaluation for seed $SEED"
    
    # Generation
    accelerate launch --config_file "$CONFIG_FILE" "$GENERATION_SCRIPT" --seed "$SEED" --method "gold" --user_id "$HF_USER_ID" --proportion -1 --k -1

    # Scoring
    accelerate launch --config_file "$CONFIG_FILE" "$SCORING_SCRIPT" --seed "$SEED" --method "gold" --user_id "$HF_USER_ID" --proportion -1 --k -1
done

# Maxmin
echo "========================="
echo "         MAXMIN          "
echo "========================="
echo ""

for SEED in "${SEEDS[@]}"; do
    for PROPORTION in "${PROPORTIONS[@]}"; do
        echo "Running MAXMIN evaluation for seed $SEED, proportion $PROPORTION"

        # Generation
        accelerate launch --config_file "$CONFIG_FILE" "$GENERATION_SCRIPT" --seed "$SEED" --proportion "$PROPORTION" --method "maxmin" --user_id "$HF_USER_ID" --k -1

        # Scoring
        accelerate launch --config_file "$CONFIG_FILE" "$SCORING_SCRIPT" --seed "$SEED" --proportion "$PROPORTION" --method "maxmin" --user_id "$HF_USER_ID" --k -1

    done
done

# SharedRep
echo "========================="
echo "        SHAREDREP        "
echo "========================="
echo ""

for SEED in "${SEEDS[@]}"; do
    for PROPORTION in "${PROPORTIONS[@]}"; do
        for K in "${Ks[@]}"; do
            echo "Running SHAREDREP evaluation for seed $SEED, proportion $PROPORTION, and K $K"

            # Generation
            accelerate launch --config_file "$CONFIG_FILE" "$GENERATION_SCRIPT" --seed "$SEED" --proportion "$PROPORTION" --k "$K" --method "sharedrep" --user_id "$HF_USER_ID"

            # Scoring
            accelerate launch --config_file "$CONFIG_FILE" "$SCORING_SCRIPT" --seed "$SEED" --proportion "$PROPORTION" --k "$K" --method "sharedrep" --user_id "$HF_USER_ID"
        done
    done
done