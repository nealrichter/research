# microgpt_sft.py — An Annotated Guide

**LoRA Supervised Fine-Tuning in Pure Python**

@nealrichter + Amazon Kiro

---

Source: [microgpt\_sft.py on GitHub](https://github.com/nealrichter/research/blob/main/microgpt/microgpt_sft.py)

---

## Overview

| Parameter | Value |
|-----------|-------|
| LoRA rank (r) | 4 |
| LoRA alpha (α) | 8 |
| Target modules | attn_wq, attn_wv |
| Trainable params | 256 (5.7% of base) |
| Epochs | 50 |

---

## Frozen Base Model

```
 93  state_dict = {k: [[Value(w) for w in row] for row in mat]
 94                for k, mat in model['weights'].items()}
 97      if len(state_dict[key]) < vocab_size:
 98          state_dict[key].append([Value(random.gauss(0, 0.08)) for _ in range(n_embd)])
101  block_size = 32
102  while len(state_dict['wpe']) < block_size:
103      state_dict['wpe'].append([Value(random.gauss(0, 0.08)) for _ in range(n_embd)])
106  base_params = [p for mat in state_dict.values() for row in mat for p in row]
```

---

## LoRA Adapters

> h = W₀ · x + (α/r) · W_up · W_down · x

W_down: random init. W_up: zeros (no perturbation at start) [1].

```
109  lora_rank = 4
110  lora_alpha = 8
111  lora_scale = lora_alpha / lora_rank
113  matrix = lambda nout, nin, std=0.08: [[Value(random.gauss(0, std))
114      for _ in range(nin)] for _ in range(nout)]
116  lora_dict = {}
117      lora_dict[f'layer{i}.attn_wq.down'] = matrix(lora_rank, n_embd)
118      lora_dict[f'layer{i}.attn_wq.up'] = [[Value(0.0) ...] ...]
119      lora_dict[f'layer{i}.attn_wv.down'] = matrix(lora_rank, n_embd)
120      lora_dict[f'layer{i}.attn_wv.up'] = [[Value(0.0) ...] ...]
122  lora_params = [p for mat in lora_dict.values() for row in mat for p in row]
```

---

## LoRA-Augmented Forward Pass

```
126  def linear(x, w):
127      return [sum(wi * xi for wi, xi in zip(wo, x)) for wo in w]
129  def linear_lora(x, w_base, w_down, w_up):
130      """h = W_base @ x + scale * W_up @ (W_down @ x)"""
131      return [b + lora_scale * l for b, l in zip(
132          linear(x, w_base), linear(linear(x, w_down), w_up))]
```

Applied to Q and V only (K and MLP unchanged) [1]:

```
142  def gpt(token_id, pos_id, keys, values):
...
148          q = linear_lora(x, state_dict[f'layer{li}.attn_wq'],
149                  lora_dict[f'layer{li}.attn_wq.down'], lora_dict[f'layer{li}.attn_wq.up'])
150          k = linear(x, state_dict[f'layer{li}.attn_wk'])
151          v = linear_lora(x, state_dict[f'layer{li}.attn_wv'],
152                  lora_dict[f'layer{li}.attn_wv.down'], lora_dict[f'layer{li}.attn_wv.up'])
```

---

## Response-Only Loss Masking

Loss only on response tokens (after SEP) [2]:

> L_SFT = -(1/|R|) · Σⱼ∈R log P(tⱼ | t₁, …, tⱼ₋₁)

```
222      for instruction, response in sft_data:
223          instr_tok = encode(instruction)
224          resp_tok = encode(response)
225          tokens = [BOS] + instr_tok + [SEP] + resp_tok + [BOS]
227          resp_start = len(instr_tok) + 2
233              if pos_id >= resp_start - 1:  # loss only on response tokens
234                  probs = softmax(logits)
235                  losses.append(-probs[tokens[pos_id + 1]].log())
239          loss = (1 / len(losses)) * sum(losses)
```

---

## LoRA Merge and Export

> W_merged = W₀ + (α/r) · W_up · W_down

```
266      for i in range(n_layer):
267          for key in (f'layer{i}.attn_wq', f'layer{i}.attn_wv'):
268              w_base = state_dict[key]
269              w_down = lora_dict[f'{key}.down']
270              w_up = lora_dict[f'{key}.up']
271              for r in range(len(w_base)):
272                  for c in range(len(w_base[0])):
273                      w_base[r][c].data += lora_scale * sum(
274                          w_up[r][k].data * w_down[k][c].data
275                          for k in range(lora_rank))
```

---

## Usage

```bash
python3 microgpt.py              # pretrain (required first)
python3 microgpt_sft.py          # LoRA SFT -> model_sft.json
python3 microgpt_sft.py -i       # inference from model_sft.json
python3 microgpt_sft.py --viz 50 # with visualization
```

---

## References

1. Hu, J.E., et al. *LoRA: Low-Rank Adaptation of Large Language Models*. arXiv:2106.09685, 2021. https://arxiv.org/abs/2106.09685
2. Wei, J., et al. *Finetuned Language Models Are Zero-Shot Learners*. ICLR, 2022. https://arxiv.org/abs/2109.01652
3. Kingma, D.P. and Ba, J. *Adam: A Method for Stochastic Optimization*. ICLR, 2015. https://arxiv.org/abs/1412.6980
