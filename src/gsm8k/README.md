# Mathematical Reasoning
**Goal**: the model is required to provide solutions to high-quality grade school math word problems. Responses are evaluated based on correctness of the solution and the multi-step arithmetic reasoning (socratic)

**Dataset**: GSM8K (Cobbe, Karl et al. (2021). “Training verifiers to solve math word problems”)

## 1. 🎛️ Oracle Socratic Scorer
We first train a reward model to mimic a *socratic* oracle scorer. To do so, we subsample 5000 samples from both the main and the socratic split of GSM8K dataset. For a given prompt, we mark the socratic responses subsampled from the dataset as `chosen`, and the main ones as `rejected`. Then, we train the model over the created dataset. To replicate this, use
```bash 
accelerate launch --config_file sharedrep-rlhf/accelerate_config.yaml sharedrep-rlhf/src/gsm8k/reward_training/socratic_reward_training.py --user_id "HUGGINGFACE_USER_ID"
``` 

## 2. 📚 Dataset
The dataset preparation consists on:
1. Prepending the following *few shot prefix* to each question, and sample 2 responses $\boldsymbol{y}_1, \boldsymbol{y}_2 \sim\pi(\cdot|\boldsymbol{x})$, i.e., from `Qwen/Qwen2.5-Math-1.5B`.
```python
FEW_SHOT_PREFIX = """These are examples of how to solve math problems step by step:

Q: If there are 3 apples and you eat one, how many are left?
A: Let's think step by step. There are 3 apples. You eat one. So 3 - 1 = 2.
#### The final answer is 2.

Q: Tom had 4 pencils. He gave 1 to Sarah and bought 3 more. How many does he have now?
A: Let's think step by step. He had 4 and gave away 1, so 4 - 1 = 3. Then he bought 3, so 3 + 3 = 6.
#### The final answer is 6.

The following is the only question you need to answer:"""
```

```bash
python sharedrep-rlhf/src/gsm8k/dataset/generate_responses.py --user_id "HF_USER_ID"
```

2. Score them using *conciseness*, *correctness*, and the *socratic* gold rewards `{HF_USER_ID}/socratic-gpt2-large` we trained before:
```bash
python sharedrep-rlhf/src/gsm8k/dataset/score_responses.py --user_id "HF_USER_ID"
```

3. Create a dataset with column names`group_id, chosen, rejected` for a given seed and a minority proportion. This will push the dataset to the hub at `{user_id}/qwen2.5-math-1.5b-gsm8k-seed{SEED}-prop{PROP}`.
```bash
python sharedrep-rlhf/src/gsm8k/dataset/create_chosen_rejected.py --user_id "HF_USER_ID" --seed SEED --proportion PROP
```


## 3. 🎯 Reward Training
Train reward models according to the following methodologies:
1. **Maxmin**:
```bash
accelerate launch --config_file path/to/config.yaml sharedrep-rlhf/src/gsm8k/reward_training/maxmin_reward_training.py --seed "$SEED" --proportion "$PROPORTION" --group "$GID" --user_id "$HF_USER_ID"
```
2. **SharedRep**:
```bash
accelerate launch --config_file path/to/config.yaml sharedrep-rlhf/src/gsm8k/reward_training/sharedrep_reward_training.py --seed "$SEED" --proportion "$PROPORTION" --k "$K" --user_id "$HF_USER_ID"
```


## 4. 🤖 Reinforcement Learning
Train a RLHF policy using GRPO using the different reward models that follow:
1. **Gold**:
```bash
accelerate launch --config_file path/to/config.yaml sharedrep-rlhf/src/gsm8k/grpo_training/gold_grpo_training.py --seed "$SEED" --user_id "$HF_USER_ID"
```
2. **Maxmin**:
```bash
accelerate launch --config_file path/to/config.yaml sharedrep-rlhf/src/gsm8k/grpo_training/maxmin_grpo_training.py --seed "$SEED" --proportion "$PROPORTION" --user_id "$HF_USER_ID"
```
3. **SharedRep**:
```bash
accelerate launch --config_file path/to/config.yaml sharedrep-rlhf/src/gsm8k/grpo_training/sharedrep_grpo_training.py --seed "$SEED" --proportion "$PROPORTION" --k "$K" --user_id "$HF_USER_ID"
```


## 5. 🎓 Evaluation
A method evaluation consists on: , and .
- Sampling a response $y$ given an input prompt $x$ from the trained policy $\pi_{\rm method}$. Use
```bash
GENERATION_SCRIPT=sharedrep-rlhf/src/gsm8k/evaluation/generate_responses.py
```
- Scoring $(x,y)$ using the gold reward models. Use
```bash
SCORING_SCRIPT=sharedrep-rlhf/src/gsm8k/evaluation/score_responses.py
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
