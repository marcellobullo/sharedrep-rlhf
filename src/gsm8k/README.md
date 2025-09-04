# GSM8K

## 1. 🎛️ Oracle socratic Scorer
We first train a reward model to mimic a *socratic* oracle scorer. To do so, we subsample 5000 samples from both the main and the socratic split of GSM8K dataset. For a given prompt, we mark the socratic responses subsampled from the dataset as `chosen`, and the main ones as `rejected`. Then, we train the model over the created dataset. To replicate this, use
``` 
accelerate launch --config_file sharedrep-rlhf/accelerate_config.yaml sharedrep-rlhf/src/gsm8k/reward_training/socratic_reward_training.py --user_id USER_ID
``` 


## 1. 🎛️ SFT


## 2. 📚 Dataset


## 3. 🎯 Reward Training


## 4. 🤖 Reinforcement Learning


## 5. 🎓 Evaluation
