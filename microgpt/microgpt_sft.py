"""
LoRA Supervised Fine-Tuning for microgpt.
Loads pretrained model from model.json, freezes base weights, trains LoRA adapters
on instruction-response pairs with response-only loss masking.
Outputs merged model to model_sft.json.

Pure Python, zero dependencies.
Based on Hu et al. "LoRA" (2021), Dettmers et al. "QLoRA" (2024).
"""

import json, math, random, sys, os
random.seed(42)

# -h/--help: print usage and exit before doing any work
if '-h' in sys.argv or '--help' in sys.argv:
    print(
        "usage: python3 microgpt_sft.py [-i [MODEL.json]] [--viz [N]] [-h]\n\n"
        "LoRA-finetune the pretrained model on instruction->response pairs, then sample.\n\n"
        "  -i [MODEL.json]  inference only from saved weights (default model_sft.json)\n"
        "  --viz [N]        loss sparkline + attention heat map; N>0 dumps every N steps\n"
        "  -h, --help       show this help and exit"
    )
    sys.exit(0)

# -i flag: inference-only mode using a saved SFT model file
inference_only = '-i' in sys.argv
# --viz [N]: ASCII visualization (loss sparkline + attention heat map); all logic in microgpt_viz.py
import microgpt_viz as viz
viz.configure(sys.argv)
viz.tee_stdout()  # mirror all stdout into an appended train.log
if inference_only:
    random.seed()
    _idx = sys.argv.index('-i')
    _nxt = sys.argv[_idx + 1] if _idx + 1 < len(sys.argv) else ''
    model_file = _nxt if _nxt and not _nxt.startswith('-') else 'model_sft.json'  # ignore following flags
    if not os.path.exists(model_file):
        print(f"error: {model_file} not found. Run `python3 microgpt_sft.py` first to train.")
        sys.exit(1)

# --- Load model ---
src_file = model_file if inference_only else 'model.json'
with open(src_file) as f:
    model = json.load(f)

uchars = model['vocab']
cfg = model['config']
n_layer, n_embd, n_head = cfg['n_layer'], cfg['n_embd'], cfg['n_head']
head_dim = n_embd // n_head
BOS = len(uchars)
SEP = len(uchars) + 1
vocab_size = len(uchars) + 2

# --- Autograd engine ---
class Value:
    __slots__ = ('data', 'grad', '_children', '_local_grads')
    def __init__(self, data, children=(), local_grads=()):
        self.data = data
        self.grad = 0
        self._children = children
        self._local_grads = local_grads
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data + other.data, (self, other), (1, 1))
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data * other.data, (self, other), (other.data, self.data))
    def __pow__(self, other): return Value(self.data**other, (self,), (other * self.data**(other-1),))
    def log(self): return Value(math.log(self.data), (self,), (1/self.data,))
    def exp(self): return Value(math.exp(self.data), (self,), (math.exp(self.data),))
    def relu(self): return Value(max(0, self.data), (self,), (float(self.data > 0),))
    def __neg__(self): return self * -1
    def __radd__(self, other): return self + other
    def __sub__(self, other): return self + (-other)
    def __rsub__(self, other): return other + (-self)
    def __rmul__(self, other): return self * other
    def __truediv__(self, other): return self * other**-1
    def __rtruediv__(self, other): return other * self**-1
    def backward(self):
        topo, visited = [], set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._children:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        self.grad = 1
        for v in reversed(topo):
            for child, lg in zip(v._children, v._local_grads):
                child.grad += lg * v.grad

# --- Rebuild state_dict from saved weights (frozen) ---
state_dict = {k: [[Value(w) for w in row] for row in mat] for k, mat in model['weights'].items()}

# Extend embedding tables for SEP token if needed
for key in ('wte', 'lm_head'):
    if len(state_dict[key]) < vocab_size:
        state_dict[key].append([Value(random.gauss(0, 0.08)) for _ in range(n_embd)])

# Extend positional embeddings for longer sequences
block_size = 32
while len(state_dict['wpe']) < block_size:
    state_dict['wpe'].append([Value(random.gauss(0, 0.08)) for _ in range(n_embd)])
block_size = len(state_dict['wpe'])

base_params = [p for mat in state_dict.values() for row in mat for p in row]

# --- LoRA adapters on Q and V projections ---
lora_rank = 4
lora_alpha = 8
lora_scale = lora_alpha / lora_rank

matrix = lambda nout, nin, std=0.08: [[Value(random.gauss(0, std)) for _ in range(nin)] for _ in range(nout)]

lora_dict = {}
for i in range(n_layer):
    lora_dict[f'layer{i}.attn_wq.down'] = matrix(lora_rank, n_embd)
    lora_dict[f'layer{i}.attn_wq.up'] = [[Value(0.0) for _ in range(lora_rank)] for _ in range(n_embd)]
    lora_dict[f'layer{i}.attn_wv.down'] = matrix(lora_rank, n_embd)
    lora_dict[f'layer{i}.attn_wv.up'] = [[Value(0.0) for _ in range(lora_rank)] for _ in range(n_embd)]

lora_params = [p for mat in lora_dict.values() for row in mat for p in row]
print(f"base params: {len(base_params)} (frozen), lora params: {len(lora_params)} (trainable)")

# --- Model forward pass ---
def linear(x, w):
    return [sum(wi * xi for wi, xi in zip(wo, x)) for wo in w]

def linear_lora(x, w_base, w_down, w_up):
    """h = W_base @ x + scale * W_up @ (W_down @ x)"""
    return [b + lora_scale * l for b, l in zip(linear(x, w_base), linear(linear(x, w_down), w_up))]

def softmax(logits):
    max_val = max(v.data for v in logits)
    exps = [(v - max_val).exp() for v in logits]
    total = sum(exps)
    return [e / total for e in exps]

def rmsnorm(x):
    return [xi * ((sum(xi * xi for xi in x) / len(x) + 1e-5) ** -0.5) for xi in x]

def gpt(token_id, pos_id, keys, values):
    x = [t + p for t, p in zip(state_dict['wte'][token_id], state_dict['wpe'][pos_id])]
    x = rmsnorm(x)
    for li in range(n_layer):
        x_res = x
        x = rmsnorm(x)
        q = linear_lora(x, state_dict[f'layer{li}.attn_wq'],
                        lora_dict[f'layer{li}.attn_wq.down'], lora_dict[f'layer{li}.attn_wq.up'])
        k = linear(x, state_dict[f'layer{li}.attn_wk'])
        v = linear_lora(x, state_dict[f'layer{li}.attn_wv'],
                        lora_dict[f'layer{li}.attn_wv.down'], lora_dict[f'layer{li}.attn_wv.up'])
        keys[li].append(k)
        values[li].append(v)
        x_attn = []
        for h in range(n_head):
            hs = h * head_dim
            q_h = q[hs:hs+head_dim]
            k_h = [ki[hs:hs+head_dim] for ki in keys[li]]
            v_h = [vi[hs:hs+head_dim] for vi in values[li]]
            attn_logits = [sum(q_h[j] * k_h[t][j] for j in range(head_dim)) / head_dim**0.5
                          for t in range(len(k_h))]
            attn_weights = softmax(attn_logits)
            if viz.enabled: viz.attn(pos_id, h, attn_weights)  # cache head-0 attention for viz
            x_attn.extend([sum(attn_weights[t] * v_h[t][j] for t in range(len(v_h)))
                          for j in range(head_dim)])
        x = [a + b for a, b in zip(linear(x_attn, state_dict[f'layer{li}.attn_wo']), x_res)]
        x_res = x
        x = [xi.relu() for xi in linear(rmsnorm(x), state_dict[f'layer{li}.mlp_fc1'])]
        x = [a + b for a, b in zip(linear(x, state_dict[f'layer{li}.mlp_fc2']), x_res)]
    return linear(x, state_dict['lm_head'])

# --- Load SFT data ---
sft_data = []
for line in open('input_sft.txt'):
    if '|' in line:
        instr, resp = line.strip().split('|', 1)
        sft_data.append((instr, resp))
print(f"loaded {len(sft_data)} sft pairs from input_sft.txt")

def encode(text):
    return [uchars.index(ch) for ch in text if ch in uchars]

# --- Inference helper ---
test_instructions = ["fa", "mj", "fs"]

def run_inference(label):
    print(f"\n--- {label} ---")
    for instruction in test_instructions:
        prompt = [BOS] + encode(instruction) + [SEP]
        keys, values = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
        for pos_id, tid in enumerate(prompt[:-1]):
            gpt(tid, pos_id, keys, values)
        tid, pos_id, out = prompt[-1], len(prompt) - 1, []
        for _ in range(20):
            probs = softmax([l / 0.5 for l in gpt(tid, pos_id, keys, values)])
            tid = random.choices(range(vocab_size), weights=[p.data for p in probs])[0]
            if tid in (BOS, SEP):
                break
            out.append(uchars[tid])
            pos_id += 1
        print(f"  Q: {instruction}  A: {''.join(out)}")

# --- Inference or Training ---
if inference_only:
    run_inference("SFT model inference")
else:
    run_inference("BEFORE SFT (base model only)")

    # --- LoRA SFT training (response-only loss) ---
    print("\n--- LoRA SFT training ---")
    lr, beta1, beta2, eps = 0.02, 0.85, 0.99, 1e-8
    m = [0.0] * len(lora_params)
    v_buf = [0.0] * len(lora_params)
    num_epochs = 50
    total_steps = num_epochs * len(sft_data)
    step = 0
    first_loss = last_loss = None  # end-to-end loss trajectory

    for epoch in range(num_epochs):
        random.shuffle(sft_data)
        for instruction, response in sft_data:
            instr_tok = encode(instruction)
            resp_tok = encode(response)
            tokens = [BOS] + instr_tok + [SEP] + resp_tok + [BOS]
            n = min(block_size, len(tokens) - 1)
            resp_start = len(instr_tok) + 2

            keys, values = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
            losses = []
            for pos_id in range(n):
                logits = gpt(tokens[pos_id], pos_id, keys, values)
                if pos_id >= resp_start - 1:  # loss only on response tokens
                    probs = softmax(logits)
                    losses.append(-probs[tokens[pos_id + 1]].log())

            if not losses:
                continue
            loss = (1 / len(losses)) * sum(losses)
            loss.backward()

            # Update only LoRA params
            lr_t = lr * (1 - step / total_steps)
            for i, p in enumerate(lora_params):
                m[i] = beta1 * m[i] + (1 - beta1) * p.grad
                v_buf[i] = beta2 * v_buf[i] + (1 - beta2) * p.grad ** 2
                p.data -= lr_t * (m[i] / (1 - beta1 ** (step + 1))) / ((v_buf[i] / (1 - beta2 ** (step + 1))) ** 0.5 + eps)
                p.grad = 0
            for p in base_params:
                p.grad = 0

            if first_loss is None: first_loss = loss.data
            last_loss = loss.data
            step += 1
            if viz.enabled:
                viz.step(step - 1, total_steps, loss.data)
            else:
                print(f"  step {step:4d}/{total_steps} | loss {loss.data:.4f}", end='\r')

    # --- After SFT ---
    if first_loss is not None:
        _pct = (last_loss - first_loss) / first_loss * 100 if first_loss else 0.0
        print(f"\nLoss (CE / response tokens) {first_loss:.4f} -> {last_loss:.4f}  ({_pct:+.1f}%)")
    run_inference("\nAFTER SFT (base + LoRA)")

    # --- Merge LoRA into base and save ---
    for i in range(n_layer):
        for key in (f'layer{i}.attn_wq', f'layer{i}.attn_wv'):
            w_base = state_dict[key]
            w_down = lora_dict[f'{key}.down']
            w_up = lora_dict[f'{key}.up']
            for r in range(len(w_base)):
                for c in range(len(w_base[0])):
                    w_base[r][c].data += lora_scale * sum(w_up[r][k].data * w_down[k][c].data for k in range(lora_rank))

    with open('model_sft.json', 'w') as f:
        json.dump({'vocab': uchars,
                   'config': {'n_layer': n_layer, 'n_embd': n_embd, 'block_size': block_size, 'n_head': n_head},
                   'weights': {k: [[p.data for p in row] for row in mat] for k, mat in state_dict.items()}}, f)
    print(f"\nsaved model_sft.json (run: python3 microgpt_sft.py -i)")

# end-of-run visualization (tall loss sparkline + attention matrix); no-op without --viz
if viz.enabled:
    viz.finish()
