"""
DPO (Direct Preference Optimization) for microgpt.
Loads SFT model from model_sft.json as reference policy, trains LoRA adapters
on preference pairs (prompt, chosen, rejected) using the DPO loss.
Outputs merged model to model_dpo.json.

Pure Python, zero dependencies.
Based on Rafailov et al. "Direct Preference Optimization" (NeurIPS 2023).

Verified by Gemini.
"""

import json, math, random, sys, os, signal
random.seed(42)

# -i flag: inference-only mode
inference_only = '-i' in sys.argv
# -n flag: max training steps
max_steps = None
if '-n' in sys.argv:
    max_steps = int(sys.argv[sys.argv.index('-n') + 1])
if inference_only:
    random.seed()
    _idx = sys.argv.index('-i')
    model_file = sys.argv[_idx + 1] if _idx + 1 < len(sys.argv) else 'model_dpo.json'
    if not os.path.exists(model_file):
        print(f"error: {model_file} not found. Run `python3 microgpt_dpo.py` first.")
        sys.exit(1)

# --- Load SFT model (serves as both reference and init for policy) ---
src_file = model_file if inference_only else 'model_sft.json'
if not os.path.exists(src_file):
    print(f"error: {src_file} not found. Run `python3 microgpt_sft.py` first.")
    sys.exit(1)
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

# --- Build policy model (trainable via LoRA) ---
state_dict = {k: [[Value(w) for w in row] for row in mat] for k, mat in model['weights'].items()}

# Ensure SEP token exists
for key in ('wte', 'lm_head'):
    if len(state_dict[key]) < vocab_size:
        state_dict[key].append([Value(random.gauss(0, 0.08)) for _ in range(n_embd)])

block_size = len(state_dict['wpe'])
base_params = [p for mat in state_dict.values() for row in mat for p in row]

# --- Reference model (frozen, plain floats for speed) ---
ref_weights = {k: [[w for w in row] for row in mat] for k, mat in model['weights'].items()}
for key in ('wte', 'lm_head'):
    if len(ref_weights[key]) < vocab_size:
        ref_weights[key].append([0.0] * n_embd)

# --- LoRA adapters (policy only) ---
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

# --- Policy forward pass (with LoRA, returns Value nodes) ---
def linear(x, w):
    return [sum(wi * xi for wi, xi in zip(wo, x)) for wo in w]

def linear_lora(x, w_base, w_down, w_up):
    return [b + lora_scale * l for b, l in zip(linear(x, w_base), linear(linear(x, w_down), w_up))]

def softmax(logits):
    max_val = max(v.data for v in logits)
    exps = [(v - max_val).exp() for v in logits]
    total = sum(exps)
    return [e / total for e in exps]

def rmsnorm(x):
    return [xi * ((sum(xi * xi for xi in x) / len(x) + 1e-5) ** -0.5) for xi in x]

def gpt_policy(token_id, pos_id, keys, values):
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
            x_attn.extend([sum(attn_weights[t] * v_h[t][j] for t in range(len(v_h)))
                          for j in range(head_dim)])
        x = [a + b for a, b in zip(linear(x_attn, state_dict[f'layer{li}.attn_wo']), x_res)]
        x_res = x
        x = [xi.relu() for xi in linear(rmsnorm(x), state_dict[f'layer{li}.mlp_fc1'])]
        x = [a + b for a, b in zip(linear(x, state_dict[f'layer{li}.mlp_fc2']), x_res)]
    return linear(x, state_dict['lm_head'])

# --- Reference forward pass (no LoRA, plain floats, no grad) ---
def linear_ref(x, w):
    return [sum(wo[j] * x[j] for j in range(len(x))) for wo in w]

def softmax_ref(logits):
    max_val = max(logits)
    exps = [math.exp(v - max_val) for v in logits]
    total = sum(exps)
    return [e / total for e in exps]

def rmsnorm_ref(x):
    scale = (sum(xi * xi for xi in x) / len(x) + 1e-5) ** -0.5
    return [xi * scale for xi in x]

def gpt_ref(token_id, pos_id, keys, values):
    x = [t + p for t, p in zip(ref_weights['wte'][token_id], ref_weights['wpe'][pos_id])]
    x = rmsnorm_ref(x)
    for li in range(n_layer):
        x_res = x
        x = rmsnorm_ref(x)
        q = linear_ref(x, ref_weights[f'layer{li}.attn_wq'])
        k = linear_ref(x, ref_weights[f'layer{li}.attn_wk'])
        v = linear_ref(x, ref_weights[f'layer{li}.attn_wv'])
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
            attn_weights = softmax_ref(attn_logits)
            x_attn.extend([sum(attn_weights[t] * v_h[t][j] for t in range(len(v_h)))
                          for j in range(head_dim)])
        x = [a + b for a, b in zip(linear_ref(x_attn, ref_weights[f'layer{li}.attn_wo']), x_res)]
        x_res = x
        x = [max(0, xi) for xi in linear_ref(rmsnorm_ref(x), ref_weights[f'layer{li}.mlp_fc1'])]
        x = [a + b for a, b in zip(linear_ref(x, ref_weights[f'layer{li}.mlp_fc2']), x_res)]
    return linear_ref(x, ref_weights['lm_head'])

# --- Compute log P(response | prompt) for a sequence ---
def encode(text):
    return [uchars.index(ch) for ch in text if ch in uchars]

def log_prob_policy(prompt_tok, resp_tok):
    """Sum of log P(resp_t | prompt, resp_<t) under policy (returns Value for grad)"""
    tokens = prompt_tok + resp_tok
    n = min(block_size, len(tokens) - 1)
    keys, values = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
    log_p = Value(0.0)
    resp_start = len(prompt_tok)
    for pos_id in range(n):
        logits = gpt_policy(tokens[pos_id], pos_id, keys, values)
        if pos_id >= resp_start - 1 and pos_id < len(tokens) - 1:
            probs = softmax(logits)
            log_p = log_p + probs[tokens[pos_id + 1]].log()
    return log_p

def log_prob_ref(prompt_tok, resp_tok):
    """Sum of log P(resp_t | prompt, resp_<t) under reference (returns float)"""
    tokens = prompt_tok + resp_tok
    n = min(block_size, len(tokens) - 1)
    keys, values = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
    log_p = 0.0
    resp_start = len(prompt_tok)
    for pos_id in range(n):
        logits = gpt_ref(tokens[pos_id], pos_id, keys, values)
        if pos_id >= resp_start - 1 and pos_id < len(tokens) - 1:
            probs = softmax_ref(logits)
            log_p += math.log(probs[tokens[pos_id + 1]] + 1e-10)
    return log_p

# --- Preference data: (prompt, chosen, rejected) ---
# Format in input_dpo.txt: prompt|chosen|rejected
dpo_data = []
if os.path.exists('input_dpo.txt'):
    for line in open('input_dpo.txt'):
        parts = line.strip().split('|')
        if len(parts) == 3:
            dpo_data.append(tuple(parts))
else:
    # Default preference pairs for teaching (names domain)
    # Chosen = correct name for the code, Rejected = wrong name
    dpo_data = [
        ("fa", "Alice", "Bob"),
        ("mb", "Benjamin", "Alice"),
        ("fc", "Charlotte", "Daniel"),
        ("md", "Daniel", "Charlotte"),
        ("fe", "Emily", "Frank"),
        ("mf", "Frank", "Emily"),
        ("fg", "Grace", "Henry"),
        ("mh", "Henry", "Grace"),
        ("fi", "Isabella", "James"),
        ("mj", "James", "Isabella"),
        ("fk", "Katherine", "Lucas"),
        ("ml", "Lucas", "Katherine"),
        ("fm", "Maria", "Nathan"),
        ("mn", "Nathan", "Maria"),
        ("fo", "Olivia", "Patrick"),
        ("mp", "Patrick", "Olivia"),
        ("fr", "Rachel", "Samuel"),
        ("ms", "Samuel", "Rachel"),
        ("ft", "Teresa", "William"),
        ("mw", "William", "Teresa"),
    ]
print(f"loaded {len(dpo_data)} preference pairs")

# --- Sigmoid helper ---
def sigmoid(x):
    """Numerically stable sigmoid for Value nodes"""
    if isinstance(x, Value):
        return Value(1.0) / (Value(1.0) + (x * -1).exp())
    return 1.0 / (1.0 + math.exp(-x))

# --- Inference helper ---
test_instructions = ["fa", "mj", "fs"]

def run_inference(label):
    print(f"\n--- {label} ---")
    for instruction in test_instructions:
        prompt = [BOS] + encode(instruction) + [SEP]
        keys, values = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
        for pos_id, tid in enumerate(prompt[:-1]):
            gpt_policy(tid, pos_id, keys, values)
        tid, pos_id, out = prompt[-1], len(prompt) - 1, []
        for _ in range(20):
            probs = softmax([l / 0.5 for l in gpt_policy(tid, pos_id, keys, values)])
            tid = random.choices(range(vocab_size), weights=[p.data for p in probs])[0]
            if tid in (BOS, SEP):
                break
            out.append(uchars[tid])
            pos_id += 1
        print(f"  Q: {instruction}  A: {''.join(out)}")

# --- Inference or Training ---
if inference_only:
    run_inference("DPO model inference")
else:
    run_inference("BEFORE DPO (SFT model)")

    # --- DPO training ---
    print("\n--- DPO training ---")
    beta = 0.3  # KL penalty strength
    lr, beta1, beta2, eps = 0.005, 0.85, 0.99, 1e-8
    m = [0.0] * len(lora_params)
    v_buf = [0.0] * len(lora_params)
    num_epochs = 30
    total_steps = num_epochs * len(dpo_data)
    if max_steps:
        total_steps = min(total_steps, max_steps)
    step = 0
    stopped = [False]

    def handle_sigint(sig, frame):
        stopped[0] = True
        print("\n  interrupted, saving model...")
    signal.signal(signal.SIGINT, handle_sigint)

    for epoch in range(num_epochs):
        if stopped[0]:
            break
        random.shuffle(dpo_data)
        for prompt_str, chosen_str, rejected_str in dpo_data:
            if stopped[0] or step >= total_steps:
                stopped[0] = True
                break
            prompt_tok = [BOS] + encode(prompt_str) + [SEP]
            chosen_tok = encode(chosen_str) + [BOS]
            rejected_tok = encode(rejected_str) + [BOS]

            # Policy log probs (differentiable)
            log_pi_chosen = log_prob_policy(prompt_tok, chosen_tok)
            log_pi_rejected = log_prob_policy(prompt_tok, rejected_tok)

            # Reference log probs (frozen, float)
            log_ref_chosen = log_prob_ref(prompt_tok, chosen_tok)
            log_ref_rejected = log_prob_ref(prompt_tok, rejected_tok)

            # DPO loss: -log σ(β * ((log π_θ(y_w|x) - log π_ref(y_w|x)) - (log π_θ(y_l|x) - log π_ref(y_l|x))))
            reward_chosen = log_pi_chosen + Value(-log_ref_chosen)
            reward_rejected = log_pi_rejected + Value(-log_ref_rejected)
            logit = (reward_chosen + (reward_rejected * -1)) * beta
            loss = (sigmoid(logit).log()) * -1

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

            step += 1
            print(f"  step {step:4d}/{total_steps} | loss {loss.data:.4f}", end='\r')

    # --- After DPO ---
    run_inference("\nAFTER DPO")

    # --- Merge LoRA into base and save ---
    for i in range(n_layer):
        for key in (f'layer{i}.attn_wq', f'layer{i}.attn_wv'):
            w_base = state_dict[key]
            w_down = lora_dict[f'{key}.down']
            w_up = lora_dict[f'{key}.up']
            for r in range(len(w_base)):
                for c in range(len(w_base[0])):
                    w_base[r][c].data += lora_scale * sum(w_up[r][k].data * w_down[k][c].data for k in range(lora_rank))

    with open('model_dpo.json', 'w') as f:
        json.dump({'vocab': uchars,
                   'config': {'n_layer': n_layer, 'n_embd': n_embd, 'block_size': block_size, 'n_head': n_head},
                   'weights': {k: [[p.data for p in row] for row in mat] for k, mat in state_dict.items()}}, f)
    print(f"\nsaved model_dpo.json (run: python3 microgpt_dpo.py -i)")
