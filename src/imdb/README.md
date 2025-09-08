# Controlled Sentiment Analysis

**Goal:** The model is required to output responses aligned to a specified sentiment, like positive, negative or neutral

**Dataset:** IMDb movie review dataset (Maas et al., 2011)

## 1. 📚 Dataset
The IMDb dataset is pre-processed as follows: for each review in the training-split, the first 2–8 tokens are extracted as the prompt, and for each of them 2 responses  $\boldsymbol{y}_1, \boldsymbol{y}_2 \sim\pi(\cdot|\boldsymbol{x})$ are sampled from `lvwerra/gpt2-imdb`.
```bash
python sharedrep-rlhf/src/imdb/dataset/generate_responses.py --user_id "HF_USER_ID"
```

2. Score them using *conciseness*, and *positiveness*, modeled by the sentiment scorer `lvwerra/distilbert-imdb`:
```bash
python sharedrep-rlhf/src/imdb/dataset/score_responses.py --user_id "HF_USER_ID"
```

3. Create a dataset with column names`group_id, chosen, rejected` for a given seed and a minority proportion. This will push the dataset to the hub at `{user_id}/gpt2-imdb-seed{SEED}-prop{PROP}`.
```bash
python sharedrep-rlhf/src/imdb/dataset/create_chosen_rejected.py --user_id "HF_USER_ID" --seed SEED --proportion PROP
```


## 3. 🎯 Reward Training
Train reward models according to the following methodologies:
1. **Maxmin**:
```bash
accelerate launch --config_file path/to/config.yaml sharedrep-rlhf/src/imdb/reward_training/maxmin_reward_training.py --seed "$SEED" --proportion "$PROPORTION" --group "$GID" --user_id "$HF_USER_ID"
```
2. **SharedRep**:
```bash
accelerate launch --config_file path/to/config.yaml sharedrep-rlhf/src/imdb/reward_training/sharedrep_reward_training.py --seed "$SEED" --proportion "$PROPORTION" --k "$K" --user_id "$HF_USER_ID"
```


## 4. 🤖 Reinforcement Learning
Train a RLHF policy using PPO using the different reward models that follow:
1. **Gold**:
```bash
accelerate launch --config_file path/to/config.yaml sharedrep-rlhf/src/imdb/ppo_training/gold_grpo_training.py --seed "$SEED" --user_id "$HF_USER_ID"
```
2. **Maxmin**:
```bash
accelerate launch --config_file path/to/config.yaml sharedrep-rlhf/src/imdb/grpo_training/maxmin_ppo_training.py --seed "$SEED" --proportion "$PROPORTION" --user_id "$HF_USER_ID"
```
3. **SharedRep**:
```bash
accelerate launch --config_file path/to/config.yaml sharedrep-rlhf/src/imdb/grpo_training/sharedrep_ppo_training.py --seed "$SEED" --proportion "$PROPORTION" --k "$K" --user_id "$HF_USER_ID"
```


## 5. 🎓 Evaluation
A method evaluation consists on: , and .
- Sampling a response $y$ given an input prompt $x$ from the trained policy $\pi_{\rm method}$. Use
```bash
GENERATION_SCRIPT=sharedrep-rlhf/src/imdb/evaluation/generate_responses.py
```
- Scoring $(x,y)$ using the gold reward models. Use
```bash
SCORING_SCRIPT=sharedrep-rlhf/src/imdb/evaluation/score_responses.py
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



