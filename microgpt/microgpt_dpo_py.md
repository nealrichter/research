# microgpt_dpo.py — An Annotated Guide

**Direct Preference Optimization in Pure Python**

@nealrichter + Amazon Kiro

---

Source: [microgpt\_dpo.py on GitHub](https://github.com/nealrichter/research/blob/main/microgpt/microgpt_dpo.py)

---

## Overview

| Parameter | Value |
|-----------|-------|
| β (KL penalty) | 0.3 |
| Reference model | Frozen SFT checkpoint |
| LoRA rank | 4 |
| Learning rate | 0.005 |
| Epochs | 30 |
| Preference pairs | 140 |

---

## The DPO Loss

DPO [1] collapses 3-stage RLHF into one loss:

> **L_DPO = -E [ log σ( β · (log π_θ(y_w|x)/π_ref(y_w|x) - log π_θ(y_l|x)/π_ref(y_l|x)) ) ]**

The implicit reward: r(x,y) = β · log(π_θ(y|x) / π_ref(y|x)) — the model *is* the reward model.

---

## Dual Model Architecture

### Policy (differentiable, with LoRA)

```
103  state_dict = {k: [[Value(w) for w in row] for row in mat]
104                for k, mat in model['weights'].items()}
```

```
152  def gpt_policy(token_id, pos_id, keys, values):
153      x = [t + p for t, p in zip(state_dict['wte'][token_id], state_dict['wpe'][pos_id])]
154      x = rmsnorm(x)
...
158          q = linear_lora(x, state_dict[f'layer{li}.attn_wq'],
159                          lora_dict[f'layer{li}.attn_wq.down'], lora_dict[f'layer{li}.attn_wq.up'])
160          k = linear(x, state_dict[f'layer{li}.attn_wk'])
161          v = linear_lora(x, state_dict[f'layer{li}.attn_wv'],
162                          lora_dict[f'layer{li}.attn_wv.down'], lora_dict[f'layer{li}.attn_wv.up'])
```

### Reference (plain floats, no grad, ~3-5× faster)

```
114  ref_weights = {k: [[w for w in row] for row in mat]
115                 for k, mat in model['weights'].items()}
```

```
184  def linear_ref(x, w):
185      return [sum(wo[j] * x[j] for j in range(len(x))) for wo in w]
197  def gpt_ref(token_id, pos_id, keys, values):
198      # identical architecture, plain float arithmetic
```

---

## Log-Probability Computation

> log π(y|x) = Σₜ log P(yₜ | x, y<t)

```
229  def log_prob_policy(prompt_tok, resp_tok):
230      """Returns Value (differentiable)"""
231      tokens = prompt_tok + resp_tok
233      keys, values = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
234      log_p = Value(0.0)
235      resp_start = len(prompt_tok)
236      for pos_id in range(n):
237          logits = gpt_policy(tokens[pos_id], pos_id, keys, values)
238          if pos_id >= resp_start - 1 and pos_id < len(tokens) - 1:
239              probs = softmax(logits)
240              log_p = log_p + probs[tokens[pos_id + 1]].log()
241      return log_p
243  def log_prob_ref(prompt_tok, resp_tok):
244      """Returns float (frozen, no grad)"""
...
254              log_p += math.log(probs[tokens[pos_id + 1]] + 1e-10)
```

---

## DPO Training Loop

```
356          log_pi_chosen = log_prob_policy(prompt_tok, chosen_tok)
357          log_pi_rejected = log_prob_policy(prompt_tok, rejected_tok)
360          log_ref_chosen = log_prob_ref(prompt_tok, chosen_tok)
361          log_ref_rejected = log_prob_ref(prompt_tok, rejected_tok)
364          # DPO loss
365          reward_chosen = log_pi_chosen + Value(-log_ref_chosen)
366          reward_rejected = log_pi_rejected + Value(-log_ref_rejected)
367          logit = (reward_chosen + (reward_rejected * -1)) * beta
368          loss = (sigmoid(logit).log()) * -1
```

---

## Merge and Export

> W_merged = W₀ + (α/r) · W_up · W_down

```
397      for i in range(n_layer):
398          for key in (f'layer{i}.attn_wq', f'layer{i}.attn_wv'):
...
403              for r in range(len(w_base)):
404                  for c in range(len(w_base[0])):
405                      w_base[r][c].data += lora_scale * sum(
406                          w_up[r][k].data * w_down[k][c].data
407                          for k in range(lora_rank))
```

---

## Why DPO Over RLHF?

| | RLHF (PPO) | DPO |
|--|--|--|
| Stages | 3 (SFT → RM → PPO) | 1 |
| Reward model | Required | Implicit |
| Stability | Sensitive | Stable |
| Memory | 4 models | 2 models |

---

## Usage

```bash
python3 microgpt.py              # pretrain
python3 microgpt_sft.py          # LoRA SFT
python3 microgpt_dpo.py          # DPO -> model_dpo.json
python3 microgpt_dpo.py -i       # inference
python3 microgpt_dpo.py -n 500   # cap at 500 steps
python3 microgpt_dpo.py --viz 100
```

---

## References

1. Rafailov, R., et al. *Direct Preference Optimization: Your Language Model is Secretly a Reward Model*. NeurIPS, 2023. https://arxiv.org/abs/2305.18290
2. Tie, G., et al. *A Survey on Post-training of Large Language Models*. arXiv:2503.06072, 2025. https://arxiv.org/abs/2503.06072
3. Hu, J.E., et al. *LoRA: Low-Rank Adaptation of Large Language Models*. arXiv:2106.09685, 2021. https://arxiv.org/abs/2106.09685
4. Ouyang, L., et al. *Training Language Models to Follow Instructions with Human Feedback*. NeurIPS, 2022. https://arxiv.org/abs/2203.02155
5. Kingma, D.P. and Ba, J. *Adam: A Method for Stochastic Optimization*. ICLR, 2015. https://arxiv.org/abs/1412.6980
