# Single-turn Dialogue
**Goal:** The model is required to provide completions in single-turn user-assistant interactions

**Datset:** Anthropic-HH (Askell et al. 2021), a benchmark for human preference modeling in single-turn assistant interactions.

## 1. 🎛️ SFT
SFT `TinyLlama/TinyLlama-1.1B-Chat-v1.0` on HH dataset. Check `sharedrep-rlhf/src/helpful_harmless/sft/sft_tinyllama_hh.sh` to get more insights on the adopted procedure.
```bash
cd path/to/sharedrep-rlhf
chmod +x ./src/helpful_harmless/sft/sft_tinyllama_hh.sh
./src/helpful_harmless/sft/sft_tinyllama_hh.sh
```

## 2. 📚 Dataset
The dataset preparation consists on:
1. Sample 10K prompts from `chosen` column of HH dataset, and sample 2 responses (completions) for each of them. Use `--disable_chat_template` to disable the chat template.
```bash
python sharedrep-rlhf/src/helpful_harmless/dataset/generate_responses.py --user_id "$HF_USER_ID"
```

2. Score them using the gold rewards `Ray2333/gpt2-large-harmless-reward_model` and `Ray2333/gpt2-large-helpful-reward_model`
```bash
python sharedrep-rlhf/src/helpful_harmless/dataset/score_responses.py --user_id "$HF_USER_ID"
```

3. Create a dataset with column names`group_id, chosen, rejected` for a given seed and a minority proportion. This will push the dataset to the hub at `{user_id}/tinyllama-hh-10K-seed{SEED}-prop{PROP}`.
```bash
python sharedrep-rlhf/src/helpful_harmless/dataset/create_chosen_rejected.py --user_id "$HF_USER_ID" --seed SEED --proportion PROP
```
**Example** (with chat template enabled):
| `group_id` | `chosen` | `rejected` |
|------------|----------|------------|
| 1| <\|user\|> How do I deal with leg cramps in the middle of the night?  <\|assistant\|> If your muscles are tensing up, you might consider taking over-the-counter pain relievers.  |  <\|user\|> How do I deal with leg cramps in the middle of the night?  <\|assistant\|> Take a hot bath with Epsom salt in it to help soothe the muscles.  |

**Full Pipeline**: `sharedrep-rlhf/src/helpful_harmless/dataset/create_dataset.sh`.


## 3. 🎯 Reward Training
Train reward models according to the following methodologies:
1. **Maxmin**:
```bash
accelerate launch --config_file path/to/config.yaml sharedrep-rlhf/src/helpful_harmless/reward_training/maxmin_reward_training.py --seed "$SEED" --proportion "$PROPORTION" --group "$GID" --user_id "$HF_USER_ID"
```
2. **SharedRep**:
```bash
accelerate launch --config_file path/to/config.yaml sharedrep-rlhf/src/helpful_harmless/reward_training/sharedrep_reward_training.py --seed "$SEED" --proportion "$PROPORTION" --k "$K" --user_id "$HF_USER_ID"
```

**Full Pipeline**: `sharedrep-rlhf/src/helpful_harmless/reward_training/reward_training.sh`.

## 4. 🤖 Reinforcement Learning
Train a RLHF policy using GRPO using the different reward models that follow:
1. **Gold**:
```bash
accelerate launch --config_file path/to/config.yaml sharedrep-rlhf/src/helpful_harmless/grpo_training/gold_grpo_training.py --seed "$SEED" --user_id "$HF_USER_ID"
```
2. **Maxmin**:
```bash
accelerate launch --config_file path/to/config.yaml sharedrep-rlhf/src/helpful_harmless/grpo_training/maxmin_grpo_training.py --seed "$SEED" --proportion "$PROPORTION" --user_id "$HF_USER_ID"
```
3. **SharedRep**:
```bash
accelerate launch --config_file path/to/config.yaml sharedrep-rlhf/src/helpful_harmless/grpo_training/sharedrep_grpo_training.py --seed "$SEED" --proportion "$PROPORTION" --k "$K" --user_id "$HF_USER_ID"
```

**Full Pipeline**: `sharedrep-rlhf/src/helpful_harmless/grpo_training/grpo_training.sh`.

## 5. 🎓 Evaluation
A method evaluation consists on: , and .
- Sampling a response $y$ given an input prompt $x$ from the trained policy $\pi_{\rm method}$. Use
```bash
GENERATION_SCRIPT=sharedrep-rlhf/src/helpful_harmless/evaluation/generate_responses.py
```
- Scoring $(x,y)$ using the gold reward models. Use
```bash
SCORING_SCRIPT=sharedrep-rlhf/src/helpful_harmless/evaluation/score_responses.py
```
1. **$\pi_{\rm gold}$**: since for $\pi_{\rm gold}$ the arguments `proportion` and `k` are not needed, a placeholder value of -1 is used.
```bash
accelerate launch --config_file path/to/config.yaml "$GENERATION_SCRIPT" --seed "$SEED" --method "gold" --user_id "$HF_USER_ID" --proportion -1 --k -1
accelerate launch --config_file path/to/config.yaml "$SCORING_SCRIPT" --seed "$SEED" --method "gold" --user_id "$HF_USER_ID" --proportion -1 --k -1
```
2. **$\pi_{\rm maxmin}$**: since for $\pi_{\rm maxmin}$ the argument `k` is not needed, a placeholder value of -1 is used.
```bash
accelerate launch --config_file path/to/config.yaml "$GENERATION_SCRIPT" --seed "$SEED" --method "maxmin" --user_id "$HF_USER_ID" --proportion "$PROPORTION" --k -1
accelerate launch --config_file path/to/config.yaml "$SCORING_SCRIPT" --seed "$SEED" --method "maxmin" --user_id "$HF_USER_ID" --proportion "$PROPORTION" --k -1
```
3. **$\pi_{\rm sharedrep}$**:
```bash
accelerate launch --config_file path/to/config.yaml "$GENERATION_SCRIPT" --seed "$SEED" --method "sharedrep" --user_id "$HF_USER_ID" --proportion "$PROPORTION" --k "$K"
accelerate launch --config_file path/to/config.yaml "$SCORING_SCRIPT" --seed "$SEED" --method "sharedrep" --user_id "$HF_USER_ID" --proportion "$PROPORTION" --k "$K"
```

**Full Pipeline**: `sharedrep-rlhf/src/helpful_harmless/evaluation/evaluation.sh`.