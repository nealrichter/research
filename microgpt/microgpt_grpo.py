"""
GRPO (Group Relative Policy Optimization) for microgpt.
Loads SFT model from model_sft.json as reference policy, trains LoRA adapters
using group-sampled completions scored by a rule-based reward function.
No critic/value network required — advantages are group-relative.
Outputs merged model to model_grpo.json.

Supports two modes:
  Online (default): generates G completions per prompt at runtime via sampling.
  Offline (--offline FILE): reads pre-generated response groups from a JSONL file,
    skipping the expensive generation step. Useful when a stronger teacher model
    has pre-generated the rollouts.

Pure Python, zero dependencies.
Based on Shao et al. "DeepSeekMath" (2024) and DeepSeek-AI "DeepSeek-R1" (2025).
"""

import json, math, random, sys, os, signal
random.seed(42)

# -h/--help: print usage and exit before doing any work
if '-h' in sys.argv or '--help' in sys.argv:
    print(
        "usage: python3 microgpt_grpo.py [-i [MODEL.json]] [--viz [N]] [-n STEPS]\n"
        "                                [--offline FILE] [-d FILE] [-h]\n\n"
        "GRPO-align the SFT model using group-relative rewards, then generate samples.\n\n"
        "  -i [MODEL.json]  inference only from saved weights (default model_grpo.json)\n"
        "  -d FILE          training data file (default input_sft.txt, online mode)\n"
        "  --offline FILE   offline mode: read pre-generated groups from JSONL file\n"
        "                   (skips runtime generation; see input_grpo_offline.jsonl)\n"
        "  --viz [N]        loss sparkline + attention heat map; N>0 dumps every N steps\n"
        "  -n STEPS         cap training to STEPS steps\n"
        "  -h, --help       show this help and exit"
    )
    sys.exit(0)

# -i flag: inference-only mode
inference_only = '-i' in sys.argv
# --offline FILE: offline mode (pre-generated groups from JSONL)
offline_mode = '--offline' in sys.argv
offline_file = None
if offline_mode:
    _idx = sys.argv.index('--offline')
    if _idx + 1 < len(sys.argv) and not sys.argv[_idx + 1].startswith('-'):
        offline_file = sys.argv[_idx + 1]
    else:
        offline_file = 'input_grpo_offline.jsonl'
    if not inference_only and not os.path.exists(offline_file):
        print(f"error: offline data file '{offline_file}' not found.")
        sys.exit(1)
# --viz [N]: ASCII visualization (loss sparkline + attention heat map); all logic in microgpt_viz.py
import microgpt_viz as viz
viz.configure(sys.argv)
# -o flag: override output log file
_log_file = sys.argv[sys.argv.index("-o") + 1] if "-o" in sys.argv else "train_microgpt_grpo.log"
viz.tee_stdout(_log_file, append=False)
# -n flag: max training steps
max_steps = None
if '-n' in sys.argv:
    max_steps = int(sys.argv[sys.argv.index('-n') + 1])
if inference_only:
    random.seed()
    _idx = sys.argv.index('-i')
    _nxt = sys.argv[_idx + 1] if _idx + 1 < len(sys.argv) else ''
    model_file = _nxt if _nxt and not _nxt.startswith('-') else 'model_grpo.json'
    if not os.path.exists(model_file):
        print(f"error: {model_file} not found. Run `python3 microgpt_grpo.py` first.")
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
            if viz.enabled: viz.attn(pos_id, h, attn_weights)
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

def generate_policy(prompt_tok, max_len=12, temperature=1.2):
    """Sample a completion from the current policy (no gradient tracking)."""
    keys, values = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
    # Feed prompt
    for pos_id, tid in enumerate(prompt_tok[:-1]):
        gpt_policy(tid, pos_id, keys, values)
    tid, pos_id = prompt_tok[-1], len(prompt_tok) - 1
    out = []
    for _ in range(max_len):
        logits = gpt_policy(tid, pos_id, keys, values)
        # Sample from softmax with temperature (use .data to avoid building graph)
        max_l = max(l.data for l in logits)
        exps = [math.exp((l.data - max_l) / temperature) for l in logits]
        total = sum(exps)
        probs = [e / total for e in exps]
        tid = random.choices(range(vocab_size), weights=probs)[0]
        if tid in (BOS, SEP):
            break
        out.append(tid)
        pos_id += 1
        if pos_id >= block_size - 1:
            break
    return out

def log_prob_policy(prompt_tok, resp_tok):
    """Sum of log P(resp_t | prompt, resp_<t) under policy (returns Value for grad)."""
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
    """Sum of log P(resp_t | prompt, resp_<t) under reference (returns float)."""
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

# --- Reward functions (rule-based, no learned reward model) ---
def reward_fn(prompt_str, response_tok):
    """
    Score a completion with a rule-based reward.
    For the names domain:
      - Length reward: smooth reward for plausible name lengths (3-8 chars)
      - Character quality: reward lowercase letters, penalize repeats
      - Exact/partial match bonus against known correct names
    Returns a float reward. Uses continuous scoring to avoid uniform groups.
    """
    response_str = ''.join(uchars[t] for t in response_tok if t < len(uchars))
    r = 0.0

    # Empty or degenerate
    if len(response_str) == 0:
        return -1.0

    # Length reward (smooth): ideal is 4-7 chars for a name
    n = len(response_str)
    if n <= 2:
        r += 0.1 * n
    elif 3 <= n <= 8:
        r += 1.0 + 0.1 * (min(n, 6) - 3)  # peaks at 6
    else:
        r += max(0, 1.0 - 0.15 * (n - 8))  # taper off

    # Character diversity: penalize repeated chars
    if n > 1:
        unique_ratio = len(set(response_str)) / n
        r += unique_ratio * 0.5

    # Letter quality: reward if mostly lowercase letters
    alpha_count = sum(1 for c in response_str if c.isalpha())
    if n > 0:
        r += (alpha_count / n) * 0.3

    # Check if it matches known correct names from SFT data
    sft_correct = {
        'fa': ['alice', 'anna', 'amelia'],
        'mb': ['benjamin', 'brian'],
        'fc': ['charlotte'],
        'md': ['daniel'],
        'fe': ['emily'],
        'mf': ['frank'],
        'fg': ['grace'],
        'mh': ['henry'],
        'fi': ['isabella'],
        'mj': ['james'],
        'fk': ['katherine'],
        'ml': ['lucas'],
        'fm': ['maria'],
        'mn': ['nathan'],
        'fo': ['olivia'],
        'mp': ['patrick'],
        'fr': ['rachel'],
        'ms': ['samuel'],
        'ft': ['teresa'],
        'mw': ['william'],
    }
    if prompt_str in sft_correct:
        best_match = 0.0
        for correct in sft_correct[prompt_str]:
            if response_str.lower() == correct:
                best_match = 3.0  # exact match
                break
            # Partial match: shared prefix length
            prefix_len = 0
            for a, b in zip(response_str.lower(), correct):
                if a == b:
                    prefix_len += 1
                else:
                    break
            if prefix_len > 0:
                best_match = max(best_match, 0.5 + prefix_len * 0.4)
        r += best_match

    return r

def reward_fn_offline(response_str, target_str):
    """
    Rule-based reward for offline mode.
    Binary scoring: +1.0 for exact match, -1.0 for wrong.
    Clean signal for GRPO — no partial credit noise.
    """
    if not response_str:
        return -1.0
    if response_str.strip() == target_str.strip():
        return 1.0
    return -1.0

# --- GRPO training data ---
# Online mode: load prompts from input_sft.txt (or defaults)
# Offline mode: load pre-generated groups from JSONL file
grpo_prompts = []
offline_data = []

if offline_mode:
    # Offline: read pre-generated response groups from JSONL
    with open(offline_file) as f:
        for line in f:
            if line.strip():
                offline_data.append(json.loads(line.strip()))
    print(f"loaded {len(offline_data)} offline groups from {offline_file}")
else:
    # Online: just prompts (responses generated at runtime)
    _data_file = sys.argv[sys.argv.index('-d') + 1] if '-d' in sys.argv else 'input_sft.txt'
    if os.path.exists(_data_file):
        for line in open(_data_file):
            parts = line.strip().split('|')
            if len(parts) >= 1:
                grpo_prompts.append(parts[0])
    else:
        grpo_prompts = ["fa", "mb", "fc", "md", "fe", "mf", "fg", "mh",
                        "fi", "mj", "fk", "ml", "fm", "mn", "fo", "mp",
                        "fr", "ms", "ft", "mw"]
    print(f"loaded {len(grpo_prompts)} prompts for GRPO (online)")

# --- GRPO hyperparameters ---
G = 8              # group size: completions per prompt (online only; offline uses file)
beta = 0.1         # KL penalty coefficient
epsilon = 0.2      # PPO clipping parameter
gen_temp = 1.2     # sampling temperature for group generation (online only)

# --- Inference helper ---
# Sample diverse prompts from training data for test
import random as _r_test
_r_test.seed(7)
if offline_mode:
    _test_pool = [rec['prompt'] for rec in offline_data]
else:
    _test_pool = grpo_prompts if grpo_prompts else ["fa", "mj", "fs", "mb", "fk"]
test_instructions = _r_test.sample(_test_pool, min(10, len(_test_pool)))

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
    run_inference("GRPO model inference")
else:
    run_inference("BEFORE GRPO (SFT model)")

    # --- GRPO training ---
    mode_label = "offline" if offline_mode else "online"
    print(f"\n--- GRPO training ({mode_label}) ---")
    if offline_mode:
        print(f"  groups={len(offline_data)}, beta={beta}, epsilon={epsilon}")
    else:
        print(f"  group_size={G}, beta={beta}, epsilon={epsilon}, temperature={gen_temp}")
    lr, beta1, beta2, eps_adam = 0.001, 0.85, 0.99, 1e-8
    m = [0.0] * len(lora_params)
    v_buf = [0.0] * len(lora_params)
    num_epochs = 5
    data_len = len(offline_data) if offline_mode else len(grpo_prompts)
    total_steps = num_epochs * data_len
    if max_steps:
        total_steps = min(total_steps, max_steps)
    step = 0
    stopped = [False]
    first_loss = last_loss = None

    def handle_sigint(sig, frame):
        stopped[0] = True
        print("\n  interrupted, saving model...")
    signal.signal(signal.SIGINT, handle_sigint)

    for epoch in range(num_epochs):
        if stopped[0]:
            break
        if offline_mode:
            random.shuffle(offline_data)
        else:
            random.shuffle(grpo_prompts)

        items = offline_data if offline_mode else grpo_prompts
        for item in items:
            if stopped[0] or step >= total_steps:
                stopped[0] = True
                break

            # --- Branch: Online vs Offline data preparation ---
            if offline_mode:
                # Offline: read pre-generated group from JSONL record
                prompt_str = item['prompt']
                target_str = item['target']
                responses_str = item['responses']
                prompt_tok = [BOS] + encode(prompt_str) + [SEP]
                group_responses = [encode(r) + [BOS] for r in responses_str]
                rewards = [reward_fn_offline(r, target_str) for r in responses_str]
                G_eff = len(group_responses)
            else:
                # Online: generate G completions from current policy
                prompt_str = item
                prompt_tok = [BOS] + encode(prompt_str) + [SEP]
                group_responses = []
                for _ in range(G):
                    resp = generate_policy(prompt_tok, max_len=12, temperature=gen_temp)
                    group_responses.append(resp)
                rewards = [reward_fn(prompt_str, resp) for resp in group_responses]
                G_eff = G

            # --- Step 3: Advantage Calculation (group-relative) ---
            mean_r = sum(rewards) / G_eff
            var_r = sum((r - mean_r) ** 2 for r in rewards) / G_eff
            std_r = (var_r + 1e-8) ** 0.5
            advantages = [(r - mean_r) / std_r for r in rewards]

            # Skip if all rewards are identical (no signal)
            if var_r < 1e-10:
                step += 1
                if viz.enabled:
                    viz.step(step - 1, total_steps, 0.0)
                else:
                    print(f"  step {step:4d}/{total_steps} | skipped (uniform rewards)", end='\r')
                continue

            # --- Step 4: Compute old log-probs (detached, for ratio) ---
            old_log_probs = []
            for resp_tok in group_responses:
                tokens = prompt_tok + resp_tok
                n = min(block_size, len(tokens) - 1)
                keys_tmp, values_tmp = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
                lp = 0.0
                resp_start = len(prompt_tok)
                for pos_id in range(n):
                    logits = gpt_policy(tokens[pos_id], pos_id, keys_tmp, values_tmp)
                    if pos_id >= resp_start - 1 and pos_id < len(tokens) - 1:
                        probs_data = softmax(logits)
                        lp += math.log(probs_data[tokens[pos_id + 1]].data + 1e-10)
                old_log_probs.append(lp)

            # --- Step 5: GRPO Loss (clipped surrogate + KL penalty) ---
            loss = Value(0.0)
            for resp_tok, adv, old_lp in zip(group_responses, advantages, old_log_probs):
                if len(resp_tok) == 0:
                    continue

                # Current policy log-prob (differentiable)
                log_pi = log_prob_policy(prompt_tok, resp_tok)

                # Reference log-prob (frozen float)
                log_ref = log_prob_ref(prompt_tok, resp_tok)

                # Probability ratio: exp(log_pi - log_pi_old)
                log_ratio = log_pi + Value(-old_lp)
                ratio = log_ratio.exp()

                # Clipped surrogate objective
                unclipped = ratio * adv

                # clip(ratio, 1-eps, 1+eps) * advantage
                # If ratio is outside bounds, use constant (stops gradient)
                if ratio.data < 1 - epsilon:
                    clipped = Value((1 - epsilon) * adv)
                elif ratio.data > 1 + epsilon:
                    clipped = Value((1 + epsilon) * adv)
                else:
                    clipped = ratio * adv

                # min(unclipped, clipped) — pessimistic bound
                surrogate = unclipped if unclipped.data < clipped.data else clipped

                # KL penalty: approximate KL(pi_theta || pi_ref)
                kl_penalty = (log_pi + Value(-log_ref)) * beta

                # Accumulate: maximize surrogate - KL -> minimize -(surrogate - KL)
                loss = loss - (surrogate - kl_penalty) * (1.0 / G_eff)

            loss.backward()

            # --- Adam update (LoRA params only) ---
            lr_t = lr * (1 - step / total_steps)
            for i, p in enumerate(lora_params):
                m[i] = beta1 * m[i] + (1 - beta1) * p.grad
                v_buf[i] = beta2 * v_buf[i] + (1 - beta2) * p.grad ** 2
                p.data -= lr_t * (m[i] / (1 - beta1 ** (step + 1))) / ((v_buf[i] / (1 - beta2 ** (step + 1))) ** 0.5 + eps_adam)
                p.grad = 0
            for p in base_params:
                p.grad = 0

            if first_loss is None: first_loss = loss.data
            last_loss = loss.data
            step += 1
            avg_reward = mean_r
            if viz.enabled:
                viz.step(step - 1, total_steps, loss.data)
            else:
                print(f"  step {step:4d}/{total_steps} | loss {loss.data:.4f} | avg_reward {avg_reward:.2f}", end='\r')

    # --- After GRPO ---
    if first_loss is not None:
        _pct = (last_loss - first_loss) / first_loss * 100 if first_loss else 0.0
        print(f"\nLoss (GRPO {mode_label}) {first_loss:.4f} -> {last_loss:.4f}  ({_pct:+.1f}%)")
    run_inference("AFTER GRPO")

    # --- Merge LoRA into base and save ---
    for i in range(n_layer):
        for key in (f'layer{i}.attn_wq', f'layer{i}.attn_wv'):
            w_base = state_dict[key]
            w_down = lora_dict[f'{key}.down']
            w_up = lora_dict[f'{key}.up']
            for r in range(len(w_base)):
                for c in range(len(w_base[0])):
                    w_base[r][c].data += lora_scale * sum(w_up[r][k].data * w_down[k][c].data for k in range(lora_rank))

    with open('model_grpo.json', 'w') as f:
        json.dump({'vocab': uchars,
                   'config': {'n_layer': n_layer, 'n_embd': n_embd, 'block_size': block_size, 'n_head': n_head},
                   'weights': {k: [[p.data for p in row] for row in mat] for k, mat in state_dict.items()}}, f)
    print(f"\nsaved model_grpo.json (run: python3 microgpt_grpo.py -i)")

# end-of-run visualization (tall loss sparkline + attention matrix); no-op without --viz
if viz.enabled:
    viz.finish()
