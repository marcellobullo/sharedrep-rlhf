# SharedRep-RLHF: A Shared Representation Approach to RLHF with Diverse Preferences
Official repository for [SharedRep-RLHF: A Shared Representation Approach to RLHF with Diverse Preferences](https://arxiv.org/abs/2509.03672)

📎 **Abstract**: Uniform-reward reinforcement learning from human feedback (RLHF), which trains a single reward model to represent the preferences of all annotators, fails to capture the diversity of opinions across sub-populations, inadvertently favoring dominant groups. The state-of-the-art, MaxMin-RLHF, addresses this by learning group-specific reward models, and by optimizing for the group receiving the minimum reward, thereby promoting fairness. However, we identify that a key limitation of MaxMin-RLHF is its poor performance when the minimum-reward group is a minority. To mitigate this drawback, we introduce a novel framework, termed *SharedRep-RLHF*. At its core, SharedRep-RLHF learns and leverages *shared traits* in annotations among various groups, in contrast to learning separate reward models across groups. We first show that MaxMin-RLHF is provably suboptimal in learning shared traits, and then quantify the sample complexity of SharedRep RLHF. Experiments across diverse natural language tasks showcase the effectiveness of ShareRep-RLHF compared to MaxMin-RLHF with a gain of up to 20% in win rate.

🖋 **Authors**: Arpan Mukherjee, Marcello Bullo, Deniz Gündüz

---

## Getting Started
1. Create a new environment, for example using conda as follows
`conda create -n rlhf python=3.11`


`pip install -r requirements.txt`

## Experiments
Given an input prompt $\boldsymbol{x}$ and an LLM policy $\pi(\cdot|\boldsymbol{x})$, we want to sample a response $\boldsymbol{y}\sim \pi(\cdot|\boldsymbol{x})$ which are aligned with some group preferences

**How do we encode preferences?**
- We consider a finite set of traits $\{\tau_1, \tau_2, \dots, \tau_T\}$ which serve as interpretable attributes characterizing a response. Each trait corresponds to a specific qualitative dimension of the response. For example, for $T=2$:
    - $\tau_1={\rm brevity}$, then $\tau_1(\boldsymbol{y}|\boldsymbol{x})$ is the brevity of response $\boldsymbol{y}$ given $\boldsymbol{x}$
    - $\tau_2={\rm positive\ sentiment}$
- We consider $U=2$ synthetic subpopulation, with different numerosity (minority and majority)
- Each subpopulation $u\in\{1,2\}$ values a convex combination of the traits according to

$${\rm score}_u(\boldsymbol{y}|\boldsymbol{x})= \sum_{i=1}^T \lambda^u_i \tau_i(\boldsymbol{y}|\boldsymbol{x}),\qquad \sum_{i=1}^T \lambda^u_i = 1, \quad  u=1,2$$

- Given 2 responses $\boldsymbol{y}_1, \boldsymbol{y}_2$ to a prompt $\boldsymbol{x}$, subpopulation $u$ prefers $\boldsymbol{y}_1$ over $\boldsymbol{y}_2$ if

$${\rm score}_u (\boldsymbol{y}_1|\boldsymbol{x}) > {\rm score}_u(\boldsymbol{y}_2|\boldsymbol{x})$$

**Our empirical study is guided by two central questions:**
1. Does SharedRep-RLHF improve *group fairness* over MaxMin-RLHF for tasks with low minority representation, as measured by the average minority score?

2. Does SharedRep-RLHF achieve a higher *win rate* on these tasks under the same conditions?

### Experiment Tasks
- [Controlled Sentiment Analysis](src/imdb/README.md)
- [Mathematical Reasoning](src/gsm8k/README.md)
- [Single-turn Dialogue](src/helpful_harmless/README.md)


## Cite us

```latex
@article{mukherjee2025sharedrep,
  title={SharedRep-RLHF: A Shared Representation Approach to RLHF with Diverse Preferences},
  author={Mukherjee, Arpan and Bullo, Marcello and G{\"u}nd{\"u}z, Deniz},
  journal={arXiv preprint arXiv:2509.03672},
  year={2025}
}
```
