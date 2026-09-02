# microgpt.py — An Annotated Guide

**Training a GPT from Scratch in Pure Python**

@karpathy (original) | @nealrichter + Amazon Kiro (annotations)

---

This document walks through `microgpt.py`, a minimal GPT implementation in ~276 lines of pure Python with zero dependencies.

Source: [microgpt.py on GitHub](https://github.com/nealrichter/research/blob/main/microgpt/microgpt.py)

---

## Overview

| Parameter | Value |
|-----------|-------|
| Layers | 1 |
| Embedding dimension | 16 |
| Attention heads | 4 |
| Context length | 16 |
| MLP hidden dimension | 64 |
| Total parameters | ~4500 |

---

## Data and Tokenization

```
 63      if not os.path.exists('input.txt'):
 64          import urllib.request
 65          names_url = 'https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt'
 66          urllib.request.urlretrieve(names_url, 'input.txt')
 67      docs = [line.strip() for line in open('input.txt') if line.strip()]
 68      random.shuffle(docs)
 72      uchars = sorted(set(''.join(docs)))
 73      BOS = len(uchars)
 74      vocab_size = len(uchars) + 1
```

> encode(c) = index(c, uchars),  decode(i) = uchars[i]

---

## Autograd Engine

```
 79  class Value:
 80      __slots__ = ('data', 'grad', '_children', '_local_grads')
 82      def __init__(self, data, children=(), local_grads=()):
 83          self.data = data
 84          self.grad = 0
 85          self._children = children
 86          self._local_grads = local_grads
 88      def __add__(self, other):
 89          other = other if isinstance(other, Value) else Value(other)
 90          return Value(self.data + other.data, (self, other), (1, 1))
 92      def __mul__(self, other):
 93          other = other if isinstance(other, Value) else Value(other)
 94          return Value(self.data * other.data, (self, other), (other.data, self.data))
 96      def __pow__(self, other):
 97          return Value(self.data**other, (self,), (other * self.data**(other-1),))
 98      def log(self): return Value(math.log(self.data), (self,), (1/self.data,))
 99      def exp(self): return Value(math.exp(self.data), (self,), (math.exp(self.data),))
100      def relu(self): return Value(max(0, self.data), (self,), (float(self.data > 0),))
```

Local gradients: ∂(a+b)/∂a=1, ∂(ab)/∂a=b, ∂(xⁿ)/∂x=nxⁿ⁻¹, ∂ln(x)/∂x=1/x, ∂eˣ/∂x=eˣ, ∂ReLU(x)/∂x=𝟙[x>0]

### Backward Pass

```
108      def backward(self):
109          topo = []
110          visited = set()
111          def build_topo(v):
112              if v not in visited:
113                  visited.add(v)
114                  for child in v._children:
115                      build_topo(child)
116                  topo.append(v)
117          build_topo(self)
118          self.grad = 1
119          for v in reversed(topo):
120              for child, local_grad in zip(v._children, v._local_grads):
121                  child.grad += local_grad * v.grad
```

> ∂L/∂cᵢ += (∂v/∂cᵢ) · (∂L/∂v)

---

## Model Architecture

GPT-2 [2] with RMSNorm [5], no biases, ReLU instead of GeLU.

```
147  def linear(x, w):
148      return [sum(wi * xi for wi, xi in zip(wo, x)) for wo in w]
150  def softmax(logits):
151      max_val = max(val.data for val in logits)
152      exps = [(val - max_val).exp() for val in logits]
153      total = sum(exps)
154      return [e / total for e in exps]
156  def rmsnorm(x):
157      ms = sum(xi * xi for xi in x) / len(x)
158      scale = (ms + 1e-5) ** -0.5
159      return [xi * scale for xi in x]
```

> softmax(zᵢ) = exp(zᵢ - max(z)) / Σⱼ exp(zⱼ - max(z))
> RMSNorm(xᵢ) = xᵢ / √(1/d · Σⱼ xⱼ² + ε)

### Transformer Forward Pass

```
161  def gpt(token_id, pos_id, keys, values):
162      tok_emb = state_dict['wte'][token_id]
163      pos_emb = state_dict['wpe'][pos_id]
164      x = [t + p for t, p in zip(tok_emb, pos_emb)]
165      x = rmsnorm(x)
167      for li in range(n_layer):
169          x_residual = x
170          x = rmsnorm(x)
171          q = linear(x, state_dict[f'layer{li}.attn_wq'])
172          k = linear(x, state_dict[f'layer{li}.attn_wk'])
173          v = linear(x, state_dict[f'layer{li}.attn_wv'])
174          keys[li].append(k)
175          values[li].append(v)
...
182              attn_logits = [sum(q_h[j] * k_h[t][j] for j in range(head_dim))
183                             / head_dim**0.5 for t in range(len(k_h))]
184              attn_weights = softmax(attn_logits)
...
187          x = linear(x_attn, state_dict[f'layer{li}.attn_wo'])
188          x = [a + b for a, b in zip(x, x_residual)]
190          x_residual = x
191          x = rmsnorm(x)
192          x = linear(x, state_dict[f'layer{li}.mlp_fc1'])
193          x = [xi.relu() for xi in x]
194          x = linear(x, state_dict[f'layer{li}.mlp_fc2'])
195          x = [a + b for a, b in zip(x, x_residual)]
197      logits = linear(x, state_dict['lm_head'])
198      return logits
```

> Attention(Q, K, V) = softmax(QKᵀ / √dₖ) · V,  dₖ = 4

---

## Training

> L = -(1/n) · Σᵢ log P(tᵢ₊₁ | t₁, …, tᵢ)

```
219          for pos_id in range(n):
220              token_id, target_id = tokens[pos_id], tokens[pos_id + 1]
221              logits = gpt(token_id, pos_id, keys, values)
222              probs = softmax(logits)
223              loss_t = -probs[target_id].log()
224              losses.append(loss_t)
225          loss = (1 / n) * sum(losses)
```

### Adam Optimizer [4]

```
231          lr_t = learning_rate * (1 - step / num_steps)
232          for i, p in enumerate(params):
233              m[i] = beta1 * m[i] + (1 - beta1) * p.grad
234              v[i] = beta2 * v[i] + (1 - beta2) * p.grad ** 2
235              m_hat = m[i] / (1 - beta1 ** (step + 1))
236              v_hat = v[i] / (1 - beta2 ** (step + 1))
237              p.data -= lr_t * m_hat / (v_hat ** 0.5 + eps_adam)
238              p.grad = 0
```

> θₜ = θₜ₋₁ - ηₜ · m̂ₜ / (√v̂ₜ + ε)

---

## Inference

```
259  temperature = 0.5
260  for sample_idx in range(20):
261      keys, values = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
262      token_id = BOS
264      for pos_id in range(block_size):
265          logits = gpt(token_id, pos_id, keys, values)
266          probs = softmax([l / temperature for l in logits])
267          token_id = random.choices(range(vocab_size),
268                                    weights=[p.data for p in probs])[0]
269          if token_id >= len(uchars):
270              break
```

> P(tᵢ) = exp(zᵢ / τ) / Σⱼ exp(zⱼ / τ)

---

## Usage

```bash
python3 microgpt.py                # pretrain -> model.json
python3 microgpt.py -i             # inference from model.json
python3 microgpt.py --viz 250      # train with visualization
python3 microgpt.py -h             # help
```

---

## References

1. Rumelhart, D.E., et al. *Learning Representations by Back-propagating Errors*. Nature, 1986.
2. Radford, A., et al. *Language Models are Unsupervised Multitask Learners*. OpenAI, 2019.
3. Vaswani, A., et al. *Attention Is All You Need*. NeurIPS, 2017. https://arxiv.org/abs/1706.03762
4. Kingma, D.P. and Ba, J. *Adam: A Method for Stochastic Optimization*. ICLR, 2015. https://arxiv.org/abs/1412.6980
5. Zhang, B. and Sennrich, R. *Root Mean Square Layer Normalization*. NeurIPS, 2019. https://arxiv.org/abs/1910.07467
6. He, K., et al. *Deep Residual Learning for Image Recognition*. CVPR, 2016. https://arxiv.org/abs/1512.03385
