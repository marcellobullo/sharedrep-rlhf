# SharedRep-RLHF: A Shared Representation Approach to RLHF with Diverse Preferences
Official repository for [SharedRep-RLHF: A Shared Representation Approach to RLHF with Diverse Preferences](https://arxiv.org/abs/2509.03672)

📎 **Abstract**: Uniform-reward reinforcement learning from human feedback (RLHF), which trains a single reward model to represent the preferences of all annotators, fails to capture the diversity of opinions across sub-populations, inadvertently favoring dominant groups. The state-of-the-art, MaxMin-RLHF, addresses this by learning group-specific reward models, and by optimizing for the group receiving the minimum reward, thereby promoting fairness. However, we identify that a key limitation of MaxMin-RLHF is its poor performance when the minimum-reward group is a minority. To mitigate this drawback, we introduce a novel framework, termed *SharedRep-RLHF*. At its core, SharedRep-RLHF learns and leverages *shared traits* in annotations among various groups, in contrast to learning separate reward models across groups. We first show that MaxMin-RLHF is provably suboptimal in learning shared traits, and then quantify the sample complexity of SharedRep RLHF. Experiments across diverse natural language tasks showcase the effectiveness of ShareRep-RLHF compared to MaxMin-RLHF with a gain of up to 20% in win rate.

🖋 **Authors**: Arpan Mukherjee, Marcello Bullo, Deniz Gündüz

---

## Getting Started
1. Create a new environment, for example using conda as follows
`conda create -n rlhf python=3.11`


`pip install -r requirements.txt`

## Tasks
- [IMDb](src/imdb/README.md)
- [GSM8K](src/gsm8k/README.md)
- [HH](src/helpful_harmless/README.md)

