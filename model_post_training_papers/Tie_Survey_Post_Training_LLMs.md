# A Survey on Post-training of Large Language Models

**Authors:** Guiyao Tie, Zeli Zhao, Dingjie Song, Fuyang Wei, Rong Zhou, Yurou Dai, Wen Yin, Zhejian Yang, Jiangyue Yan, Yao Su, Zhenhan Dai, Yifeng Xie, Yihan Cao, Lichao Sun, Pan Zhou, Lifang He, Hechang Chen, Yu Zhang, Qingsong Wen, Tianming Liu, Neil Zhenqiang Gong, Jiliang Tang, Caiming Xiong, Heng Ji, Philip S. Yu, Jianfeng Gao
**Year/Venue:** March 2025, arXiv (2503.06072v3)
**Link:** https://arxiv.org/abs/2503.06072

## 1. The Core (Key Ideas, Concepts, & Results)

* **The Problem:** Pre-trained LLMs possess broad linguistic competence but suffer from restricted reasoning capacities, ethical uncertainties, suboptimal domain-specific performance, and inability to adapt to new tasks without retraining. Previous surveys have focused on narrow subtopics (preference alignment, PEFT, etc.) but no comprehensive survey has systematically organized the full post-training landscape—from ChatGPT's RLHF to DeepSeek-R1's reasoning breakthroughs.

* **The Core Innovation:** This is the *first comprehensive survey* of Post-training Language Models (PoLMs), introducing a structured taxonomy that classifies post-training into five paradigms—Fine-tuning, Alignment, Reasoning, Efficiency, and Integration/Adaptation—while tracing the historical evolution from 2018 (BERT/GPT) through 2025 (DeepSeek-R1). The key intellectual contribution is unifying these disparate threads into a coherent framework that shows how they interrelate and build upon each other.

* **Key Equations/Formalisms:**

  1. **GRPO Objective (DeepSeek-R1's core algorithm):**
     ```
     J_GRPO(θ) = E[q~P(Q), {o_i}~π_θ_old(O|q)]
       (1/G) Σ (1/|o_i|) Σ { min[r_t(θ)·Â_i,t, clip(r_t(θ), 1-ε, 1+ε)·Â_i,t] - β·D_KL[π_θ ∥ π_ref] }
     ```
     GRPO eliminates the critic model entirely, estimating baselines from group-relative scores. This reduces training resources significantly compared to PPO while maintaining stable policy updates.

  2. **DPO Loss (the key simplification over RLHF):**
     ```
     L_DPO(π_θ; π_ref) = -E_{(x,y_w,y_l)~D}[log σ(β·log(π_θ(y_w|x)/π_ref(y_w|x)) - β·log(π_θ(y_l|x)/π_ref(y_l|x)))]
     ```
     DPO collapses the 3-stage RLHF pipeline (SFT → Reward Model → PPO) into a single-stage optimization directly on preference data, bypassing explicit reward modeling entirely.

* **Primary Results:** This is a survey paper, so results are organizational/analytical rather than empirical. The key deliverables are:
  - A complete taxonomy (Fig. 2) covering 5 paradigms, 7 dataset types, and 3 application domains
  - Historical synthesis from BERT (2018) → DeepSeek-R1 (2025) showing the progression of techniques
  - Identification of 7 open problems with feasible research directions
  - Comprehensive tables cataloging alignment methods (Table 2, 22 methods), quantization methods (Table 5, 30+ methods), self-refine methods (Table 4, 30+ methods), knowledge distillation approaches (Table 6, 30+ methods), and multi-modal models (Table 7, 40+ models)

## 2. Anomalies & Surprises (Unexpected Results)

* **Counterintuitive Findings:**
  - **DeepSeek-R1-Zero succeeds without SFT:** The model develops sophisticated reasoning (self-verification, reflection, extended CoT) purely through RL from a base model with zero supervised fine-tuning. This challenges the conventional wisdom that SFT is a prerequisite for reasoning capability. The "cold-start" RL approach demonstrates that reasoning can emerge from reward signals alone.
  - **RLAIF matches or exceeds RLHF:** AI-generated feedback achieves comparable or superior performance to human feedback (per empirical studies by Lee et al.), even when the AI labeler is the *same scale* as the policy model. This undermines the assumption that human feedback provides irreplaceable signal quality.
  - **Direct distillation outperforms RL on small models:** DeepSeek-R1's distillation of reasoning patterns into smaller models (Qwen, LLaMA variants) yields *superior* results compared to applying RL directly to those small models. The capacity-limited models can't discover reasoning through RL but can replicate it when shown examples.

* **Ablation Surprises:**
  - In GRPO, removing the critic model (a major component of PPO) doesn't degrade performance—it actually helps by reducing noise from value function estimation errors while cutting compute costs.
  - For DPO variants, SimPO shows that eliminating the reference policy entirely (a seemingly critical constraint) still yields strong alignment, suggesting the reference policy's role is more about regularization than information.

* **Edge Cases:**
  - The paper openly notes that RL-driven reasoning models struggle with "open-ended tasks" like scientific hypothesis generation—precisely the kind of creative reasoning that would be most valuable.
  - DeepSeek-R1-Zero exhibits "readable" reasoning but sometimes produces language mixing and repetitive patterns during early training, suggesting RL can find degenerate local optima before eventually escaping them.

## 3. The "Defense" (Comprehension Stress-Test)

* **Methodological Justification:** The authors chose to organize around *paradigms* (Fine-tuning, Alignment, Reasoning, Efficiency, Integration) rather than chronology or model family. This is justified because techniques cross model boundaries (GRPO originated in DeepSeekMath but applies broadly), but it creates artificial separations—e.g., Reinforcement Fine-Tuning (§3.3) vs. RL for Alignment (§4.1) vs. RL for Reasoning (§5.2) are the same underlying framework applied to different objectives.

* **The "So What?" Question:** This survey is a *map*, not a destination. Its value lies in:
  1. Showing practitioners that the post-training space is far broader than "just do RLHF/DPO" — there are 5 paradigms with dozens of techniques each
  2. Revealing that DeepSeek-R1's innovations (GRPO, cold-start RL, reasoning distillation) represent a genuine paradigm shift, not incremental improvement
  3. Identifying that the field is converging on a pattern: base model → SFT → RL alignment → reasoning RL → distillation to smaller models

  However, it does NOT provide empirical head-to-head comparisons or clear guidance on *which technique to use when*—a significant gap for practitioners.

* **Vulnerabilities:**
  - The survey is heavily biased toward DeepSeek's work (highlighted in blue throughout Fig. 1, dedicated subsections). While DeepSeek-R1 is genuinely significant, the paper reads partly as advocacy for a specific research lineage.
  - Baselines in the reasoning section are underspecified—claims about DeepSeek-R1-Zero's emergent capabilities aren't quantitatively benchmarked in this survey.
  - The "Open Problems" section (§10) is aspirational but vague on feasibility. "Develop multi-objective RL systems" is a reasonable direction but not a concrete research agenda.
  - The paper conflates *methods papers* being surveyed with the *survey's own contributions*. Many "findings" reported are just descriptions of what others did.

* **Mechanism of Action:** The paper's implicit theoretical claim is that **RL provides a stronger learning signal than supervised data for complex reasoning tasks** because it allows exploration of multiple reasoning paths rather than memorizing single annotated trajectories. This is supported by ReFT results showing RL fine-tuning outperforms SFT even on the *same data*, and by DeepSeek-R1-Zero's emergent reasoning. The mechanism is: RL's exploration → diverse CoT generation → reward selection of valid paths → policy improvement toward robust reasoning strategies.

## 4. Trajectory (Curiosity & Next Steps)

* **Implementation Ideas:**
  - **GRPO for custom reasoning tasks:** The group-relative advantage estimation (no critic needed) makes it feasible to apply RL to specialized reasoning domains with verifiable outcomes (code generation, mathematical proofs, structured data extraction) without the infrastructure overhead of training a separate value network.
  - **Reasoning distillation pipeline:** DeepSeek-R1's approach of generating ~800K reasoning traces from a large model and distilling into smaller models via SFT is directly replicable. The key insight: use 75% reasoning data + 25% non-reasoning data for the distillation mix.
  - **Model merging for multi-task adaptation:** TIES-merging (keep top 20% parameters by magnitude, resolve sign conflicts) and DARE (probabilistic delta adjustment) offer zero-cost methods to combine fine-tuned models without retraining—immediately applicable for combining domain-specific LoRA adapters.

* **Missing Links:**
  - **No coverage of test-time compute scaling:** The survey barely mentions the "scaling inference compute" paradigm (thinking longer = better answers) that makes o1/R1 powerful in practice. How much of the reasoning gain comes from the RL training vs. simply allowing more tokens at inference?
  - **Evaluation frameworks are absent:** The paper surveys techniques but never addresses how to *measure* whether post-training actually improved reasoning vs. just pattern-matching on benchmarks.
  - **Safety-capability tradeoff is hand-waved:** The tension between alignment (constraining outputs) and reasoning (exploring freely) is mentioned as an open problem but never deeply analyzed. How does o1's "cautious" alignment interact with its reasoning capability?
  - **Continual learning / catastrophic forgetting during post-training:** The paper notes models "struggle to adapt to new tasks without retraining" but doesn't deeply examine how sequential post-training stages (SFT → RLHF → reasoning RL) interact destructively.

* **Rabbit Holes:**
  - DeepSeekMath paper (Shao et al., 2024) — the origin of GRPO and its mathematical formulation
  - Rafailov et al. (2024) "From r to Q*" — reformulating DPO as a bandit problem with token-level MDP, potentially important for understanding *why* DPO works
  - The "cold-start RL" concept — need to read DeepSeek-R1 technical report in full to understand what "self-verification and reflection emerge" means mechanistically
  - Weak-to-Strong Generalization (Burns et al., 2023) — the framework for understanding how weaker supervisors can train stronger models, directly relevant to RLAIF's success
  - Task Arithmetic (Ilharco et al., 2022) — the surprising finding that you can add/subtract model capabilities via weight arithmetic, foundational to model merging

---

## 5. The Post-Training Sequence & Fine-Tuning vs. Alignment

### Canonical Post-Training Pipeline

The paper establishes a clear sequential workflow that modern LLMs follow after pre-training:

```
Pre-trained Base Model
    │
    ▼
[1] Supervised Fine-Tuning (SFT)
    │   - Adapt to instruction-following using labeled (instruction, response) pairs
    │   - Cross-entropy loss on task-specific data
    │   - Establishes baseline task competence
    │
    ▼
[2] Alignment (RLHF / RLAIF / DPO)
    │   - Align outputs with human preferences and safety constraints
    │   - Reward modeling + policy optimization (or direct preference optimization)
    │   - Makes the model helpful, harmless, and honest
    │
    ▼
[3] Reasoning Enhancement (RL for Reasoning)
    │   - Large-scale RL (GRPO) to develop multi-step inference
    │   - Chain-of-Thought emerges through reward-driven exploration
    │   - DeepSeek-R1-Zero shows this can even replace step [1]
    │
    ▼
[4] Distillation (optional)
        - Transfer reasoning capabilities to smaller models via SFT on teacher outputs
        - ~800K samples (600K reasoning + 200K non-reasoning)
```

DeepSeek-R1's actual pipeline was: Base → SFT (cold start data) → RL reasoning → Rejection sampling to generate new SFT data → Final RL alignment pass. DeepSeek-R1-Zero skipped SFT entirely, going straight from base model to RL.

### Fine-Tuning vs. Alignment: Distinct Roles

| Dimension | Fine-Tuning (§3) | Alignment (§4) |
|-----------|-------------------|----------------|
| **Goal** | Task competence — make the model *capable* | Value alignment — make the model *safe and preferred* |
| **Signal** | Ground-truth labels (supervised) | Human/AI preferences (comparative) |
| **What changes** | The model learns *what* to output | The model learns *which* outputs humans prefer |
| **Loss function** | Cross-entropy against correct answers | Reward maximization with KL constraint against reference policy |
| **Failure mode** | Underfitting (can't do the task) | Misalignment (does the task but in harmful/unhelpful ways) |
| **Data format** | (instruction, correct_response) pairs | (instruction, preferred_response, rejected_response) triples |
| **Relationship to base model** | Adds new capabilities | Constrains/redirects existing capabilities |

### Key Insight: They Solve Different Problems

**Fine-tuning** answers: *"Can the model follow instructions and produce formatted outputs at all?"* Without SFT, a base model just does next-token prediction — it won't respond to questions in a dialogue format, won't follow instructions, and won't produce structured outputs. SFT teaches the *format* and *task mechanics*.

**Alignment** answers: *"Among all possible correct outputs, which ones do humans actually want?"* A model after SFT can follow instructions but may be verbose, toxic, unhelpful, or hallucinate confidently. Alignment uses preference signals to navigate the space of competent outputs toward those that match human values.

### The Sequence Matters

The paper makes clear why the ordering is SFT → Alignment → Reasoning:

1. **SFT must come first** (in the standard pipeline) because alignment methods (RLHF/DPO) need the model to already produce coherent, instruction-following outputs. You can't express preferences over gibberish. The reference policy π_ref in DPO's formulation is typically the SFT checkpoint.

2. **Alignment and reasoning interleave in practice.** The canonical ordering suggests alignment before reasoning RL, but DeepSeek-R1's actual pipeline does reasoning RL *first*, then a *second* alignment pass afterward (SFT cold start → reasoning RL → rejection sampling → final RL alignment). The concern with unconstrained reasoning RL is reward hacking and unsafe chains, which is why DeepSeek-R1-Zero exhibited degenerate behaviors. The practical pattern is: minimal alignment → reasoning RL → final alignment cleanup.

3. **The DeepSeek-R1-Zero exception** is remarkable precisely because it violates this ordering — showing that with a sufficiently well-designed reward (rule-based accuracy + format rewards) and enough compute, reasoning can emerge directly from RL on a base model without SFT or alignment as prerequisites. However, the resulting model had readability issues (language mixing, repetition), which is why the full DeepSeek-R1 pipeline still uses SFT as a cold start.

### Adaptive Fine-Tuning as a Lightweight Alternative

The paper also distinguishes *full* fine-tuning from *adaptive* fine-tuning techniques that modify model behavior without updating all parameters:

- **Instruction Tuning** — SFT on diverse instruction datasets (Flan, Super-NatInst) for broad task generalization
- **Prefix-Tuning** — Learnable vectors prepended at each Transformer layer (frozen base model)
- **Prompt-Tuning** — Learnable vectors only at the input embedding layer (even more parameter-efficient)
- **Reinforcement Fine-Tuning (ReFT)** — Hybrid: SFT warmup then RL exploration of multiple CoT paths on the same data

These exist on a spectrum from "change everything" (full-parameter SFT) to "change almost nothing" (prompt-tuning), with the tradeoff being: more parameter updates = better task performance but higher compute cost and catastrophic forgetting risk.

---

## 6. Practical Blueprint: Building a Fast, Efficient Post-Training Pipeline Today

*Synthesizing the survey's findings into an opinionated pipeline optimized for speed, efficiency, and skilled outputs.*

### Design Principles

1. **Do less, better.** High-quality data at every stage beats volume. IFD screening (Eq. 10) and quality thresholds eliminate noise before it pollutes training.
2. **Never update what you don't need to.** LoRA/QLoRA for SFT, GRPO instead of PPO (no critic network), DPO instead of RLHF (no reward model).
3. **Steal reasoning, don't grow it.** Distillation from a reasoning-capable teacher is cheaper and better than RL on small/medium models.
4. **Validate cheaply and often.** Rule-based rewards (accuracy, format compliance) over learned reward models wherever possible.

### The Pipeline

```
┌─────────────────────────────────────────────────────────────────────────┐
│  STAGE 0: Model & Data Selection                                         │
├─────────────────────────────────────────────────────────────────────────┤
│  • Base model: Qwen2.5 or LLaMA-3 (open, strong base, active community)│
│  • Quantize base to 4-bit (QLoRA-ready) for all fine-tuning stages      │
│  • Curate domain corpus: filter with IFD scoring, deduplicate w/ ROUGE-L│
│  • Target: 10K-50K high-quality (instruction, response) pairs            │
│  • Mix: 70% domain-specific + 30% general capability (prevent forgetting)│
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  STAGE 1: QLoRA SFT (Hours, not days)                                    │
├─────────────────────────────────────────────────────────────────────────┤
│  • Method: LoRA rank 16-64 on attention Q/V projections + FFN down_proj  │
│  • 4-bit NF4 quantization of base weights (QLoRA)                        │
│  • 2-3 epochs, cosine LR schedule, bf16 adapters                         │
│  • Hardware: Single A100/H100 (7-14B models) or 4×A100 (70B)            │
│  • Key: Stop early. SFT's job is format + basic competence, not mastery. │
│  • Validate: Held-out instruction-following accuracy > 85%               │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  STAGE 2: DPO Alignment (Skip reward model entirely)                     │
├─────────────────────────────────────────────────────────────────────────┤
│  • Generate 5-10 responses per prompt from SFT model (temperature 0.7-1) │
│  • Score with: GPT-4/Claude as judge (RLAIF) OR rule-based criteria      │
│  • Construct preference pairs: (prompt, chosen, rejected)                │
│  • Train with DPO loss, β=0.1-0.5, 1 epoch                              │
│  • Reference policy = SFT checkpoint (frozen)                            │
│  • Alternative: SimPO if you want to skip even the reference model       │
│  • Total preference data needed: 5K-20K pairs                            │
│  • Hardware: Same as Stage 1 (LoRA on the same adapters)                 │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  STAGE 3: Reasoning via Distillation (Not RL)                            │
├─────────────────────────────────────────────────────────────────────────┤
│  • Teacher: DeepSeek-R1 / Qwen-QwQ / Claude (via API)                    │
│  • Generate CoT traces for your domain problems (math, code, analysis)   │
│  • Format: <think>reasoning chain</think><answer>final answer</answer>   │
│  • SFT on reasoning traces: 10K-50K examples                             │
│  • This is dramatically cheaper than running GRPO yourself                │
│  • Key insight from paper: distillation > direct RL for models < 70B     │
│  • If you DO need RL: use GRPO with rule-based rewards (no critic)       │
│     - Reward = accuracy_score + format_compliance - length_penalty        │
│     - Group size G=8-16 samples per question                             │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  STAGE 4: Merge & Deploy                                                 │
├─────────────────────────────────────────────────────────────────────────┤
│  • Merge LoRA adapters back into base model                              │
│  • If multiple skill adapters: TIES-merging (top 20% by magnitude)       │
│  • Post-training quantization: GPTQ/AWQ to 4-bit for inference           │
│  • KV-cache quantization (KIVI 2-bit) for long-context serving           │
│  • Validate on held-out domain benchmarks + safety eval                  │
└─────────────────────────────────────────────────────────────────────────┘
```

### Why This Works (Mapping to Survey Findings)

| Decision | Rationale from Paper |
|----------|---------------------|
| QLoRA over full fine-tuning | §6.2.3: LoRA matches full fine-tuning at <1% parameters. LOMO (§3.1.3) shows full FT needs 100GB+ VRAM for 65B models. |
| DPO over RLHF | §4.3: DPO eliminates reward model fitting + PPO instability. Single-stage, same theoretical guarantees. |
| RLAIF for preference labels | §4.2: AI feedback matches human feedback quality while being 10-100× cheaper and infinitely scalable. |
| Distillation over RL for reasoning | §6.3: DeepSeek-R1's key finding — distilled small models outperform same-size models trained with direct RL. |
| GRPO if you must do RL | §2.2.4: Eliminates critic model entirely, "significantly reduces training resource consumption compared to PPO." |
| TIES-merging for multi-skill | §7.3.3: Zero-cost combination of fine-tuned adapters, resolves parameter interference via sign consensus. |
| 4-bit quantization at inference | §6.1.1: AWQ/GPTQ achieve <0.5 perplexity difference with 3× speedup on 70B models. |

### Time & Cost Estimate (7-14B Model)

| Stage | Time | Hardware | Data Needed |
|-------|------|----------|-------------|
| Data curation | 1-3 days | CPU + API calls | Source corpus |
| QLoRA SFT | 2-6 hours | 1× A100 80GB | 10-50K pairs |
| DPO Alignment | 1-3 hours | 1× A100 80GB | 5-20K preference pairs |
| Reasoning Distillation | 4-12 hours | 1× A100 80GB | 10-50K CoT traces |
| Quantization + Merge | 30 min | 1× A100 80GB | — |
| **Total** | **2-5 days** | **1 GPU** | **25-120K examples** |

### What to Skip

- **Full-parameter fine-tuning** — unless you're training a foundation model or have unlimited compute
- **PPO-based RLHF** — complex, unstable, requires 2-4× the models (policy, reference, reward, value)
- **Training your own reward model** — use LLM-as-judge or rule-based rewards instead
- **Large-scale RL for reasoning** on models <70B — distill from a larger model instead
- **Prefix-tuning / Prompt-tuning** — LoRA dominates on performance-per-parameter; these are legacy

### When to Break the Rules

- **You have a novel domain with no teacher model** → You need GRPO with rule-based rewards (can't distill what doesn't exist)
- **Safety-critical deployment** → Add Constitutional AI / red-teaming loop after DPO, don't skip alignment
- **70B+ model with cluster access** → Full-parameter SFT + GRPO becomes viable and may outperform LoRA+distillation
- **You need multi-modal capabilities** → Add modal encoder (SigLIP/CLIP) + projection layer training before SFT (§7.1)
- **Knowledge needs to stay current** → Add RAG layer at inference rather than baking knowledge into weights (§7.2.2)

---

## Appendix: Post-Training Methods Reference

### A.1 Supervised Fine-Tuning (SFT)

Adjusts all (or a subset of) model parameters on labeled instruction-response pairs using cross-entropy loss. The model learns to produce correct outputs given instructions.

**Data structure:**
```json
{
  "instruction": "Summarize the following article in 2 sentences.",
  "input": "The Federal Reserve raised interest rates by 25 basis points on Wednesday, marking the tenth consecutive increase...",
  "output": "The Federal Reserve implemented its tenth consecutive rate hike, raising rates by 25 basis points. This decision reflects ongoing efforts to combat persistent inflation despite signs of economic slowdown."
}
```

---

### A.2 Instruction Tuning

A specialized form of SFT where diverse NLP tasks are reformulated as natural language instructions. The goal is broad generalization across unseen tasks, not mastery of one.

**Data structure (Flan/Super-NatInst style):**
```json
{
  "task_name": "sentiment_classification",
  "definition": "Given a product review, classify the sentiment as positive, negative, or neutral.",
  "input": "This laptop exceeded my expectations. Battery life is incredible.",
  "output": "positive",
  "examples": [
    {"input": "Terrible quality, broke after one week.", "output": "negative"},
    {"input": "It works fine, nothing special.", "output": "neutral"}
  ]
}
```

---

### A.3 Prefix-Tuning

Prepends learnable continuous vectors (prefixes) to every Transformer layer's key/value matrices. Base model weights are frozen; only prefix parameters train.

**Data structure:** Same as SFT — the method differs in *what parameters update*, not data format.

```json
{
  "instruction": "Translate English to French.",
  "input": "The weather is beautiful today.",
  "output": "Le temps est magnifique aujourd'hui."
}
```

**Config:**
```json
{
  "method": "prefix_tuning",
  "prefix_length": 20,
  "num_layers": 32,
  "trainable_params": "prefix_vectors_only",
  "reparameterization": "mlp"
}
```

---

### A.4 Prompt-Tuning

Prepends learnable soft tokens *only at the input embedding layer* (unlike prefix-tuning which operates at every layer). Even more parameter-efficient.

**Data structure:** Same as SFT input format. The soft prompt is a learned tensor, not data.

```python
# Conceptually:
soft_prompt = nn.Parameter(torch.randn(num_tokens, embedding_dim))  # e.g., shape [20, 4096]
input_embeds = torch.cat([soft_prompt, token_embeddings(input_ids)], dim=1)
# Only soft_prompt is trained; model weights frozen
```

---

### A.5 Reinforcement Fine-Tuning (ReFT)

SFT warmup followed by RL (PPO) on the *same dataset*. The model generates multiple CoT paths per problem and receives rewards for correct final answers.

**Data structure:**
```json
{
  "problem": "If a train travels 120 miles in 2 hours, what is its speed in mph?",
  "answer": "60",
  "sft_cot": "Speed = distance / time = 120 / 2 = 60 mph",
  "rl_generated_cots": [
    {"cot": "120 miles in 2 hours. 120/2 = 60 mph.", "correct": true, "reward": 1.0},
    {"cot": "2 hours for 120 miles. 120*2 = 240 mph.", "correct": false, "reward": -1.0}
  ]
}
```

---

### A.6 RLHF (Reinforcement Learning from Human Feedback)

Three-stage process: (1) SFT, (2) train a reward model on human preference comparisons, (3) optimize policy with PPO against the reward model with a KL penalty to stay near the SFT checkpoint.

**Reward model training data:**
```json
{
  "prompt": "Explain quantum entanglement to a 10-year-old.",
  "chosen": "Imagine you have two magic dice. No matter how far apart they are, when you roll one and get a 6, the other one instantly becomes a 6 too...",
  "rejected": "Quantum entanglement is a phenomenon in quantum mechanics where two particles become correlated such that the quantum state of one particle is dependent on the state of the other, regardless of spatial separation..."
}
```

**PPO training loop (conceptual):**
```
for batch in prompts:
    responses = policy.generate(batch)
    rewards = reward_model(batch, responses)
    penalized_rewards = rewards - β * KL(policy || reference_policy)
    policy.update(PPO_loss(penalized_rewards))
```

---

### A.7 RLAIF (Reinforcement Learning from AI Feedback)

Same as RLHF but human annotators are replaced by an LLM judge. The AI generates preference labels or scores, which train the reward model (or are used directly in DPO).

**AI feedback generation:**
```json
{
  "prompt": "Write a haiku about machine learning.",
  "response_a": "Neurons learn and grow / Patterns emerge from the noise / Machines start to see",
  "response_b": "Data in, model out / Gradient descent finds the way / Loss function drops low",
  "ai_judge_prompt": "Which response is a better haiku? Consider syllable count (5-7-5), imagery, and poetic quality. Respond with A or B.",
  "ai_verdict": "A",
  "ai_rationale": "Response A maintains proper 5-7-5 syllable structure and uses more evocative imagery..."
}
```

---

### A.8 DPO (Direct Preference Optimization)

Eliminates the reward model entirely. Directly optimizes the policy on preference pairs using a closed-form loss derived from the RLHF objective.

**Data structure (identical to RLHF reward model data):**
```json
{
  "prompt": "What should I do if I find a lost wallet?",
  "chosen": "You should try to return it to the owner. Check for ID inside, and if you can't find the owner, turn it in to local authorities or the nearest lost-and-found.",
  "rejected": "Finders keepers! Take the cash and toss the wallet. Nobody will know it was you."
}
```

**Training:** Single-stage. Reference model π_ref = frozen SFT checkpoint. No reward model, no PPO.

---

### A.9 GRPO (Group Relative Policy Optimization)

DeepSeek's PPO variant that eliminates the critic/value network. For each prompt, samples a *group* of G outputs, scores them all, and computes advantages *relative to the group mean*.

**Data structure (training loop):**
```json
{
  "prompt": "What is 7 × 8?",
  "group_outputs": [
    {"response": "<think>7×8 = 56</think>\n56", "reward": 1.0},
    {"response": "<think>7×8 = 54</think>\n54", "reward": 0.0},
    {"response": "<think>7+8 = 15</think>\n15", "reward": 0.0},
    {"response": "<think>7×8... 7×7=49, +7=56</think>\n56", "reward": 1.0}
  ],
  "group_mean_reward": 0.5,
  "advantages": [0.5, -0.5, -0.5, 0.5]
}
```

**Key difference from PPO:** No value network V(s) needed. Advantage = (reward_i - mean(group_rewards)) / std(group_rewards). Note: GRPO still uses the clipped surrogate objective (like PPO) and a KL penalty against a reference policy — it only eliminates the *critic*, not the rest of the PPO machinery.

---

### A.10 LoRA (Low-Rank Adaptation)

Freezes base model, adds trainable low-rank matrices (A, B) in parallel to existing weight matrices. Output becomes W·x + α·B·A·x where A∈ℝ^(r×d), B∈ℝ^(d×r), r << d.

**Config:**
```json
{
  "method": "lora",
  "r": 16,
  "alpha": 32,
  "target_modules": ["q_proj", "v_proj", "down_proj"],
  "dropout": 0.05,
  "quantization": "nf4"
}
```

**Data structure:** Same as SFT — LoRA is an *optimization method*, not a data format. Any SFT/DPO/RLHF task can use LoRA.

---

### A.11 QLoRA (Quantized LoRA)

LoRA applied on top of a 4-bit quantized base model. Base weights stored in NF4 format; LoRA adapters trained in bf16/fp16. Enables fine-tuning 70B models on a single GPU.

**Same data as LoRA.** The difference is purely in memory:
```
Full fine-tuning 70B: ~280 GB VRAM (impossible on single GPU)
LoRA 70B:             ~160 GB VRAM (needs 2× A100 80GB for base weights in fp16 + adapter optimizer states)
QLoRA 70B:            ~48 GB VRAM  (fits on 1× A100 80GB)
```

---

### A.12 Knowledge Distillation

Transfers capabilities from a large teacher model to a smaller student model. The student trains on teacher-generated outputs, matching the teacher's soft probability distribution.

**Data structure (black-box distillation / reasoning distillation):**
```json
{
  "prompt": "Prove that the square root of 2 is irrational.",
  "teacher_model": "deepseek-r1",
  "teacher_response": "<think>\nAssume √2 is rational, so √2 = p/q where p,q are integers with no common factors.\nThen 2 = p²/q², so p² = 2q².\nThis means p² is even, so p must be even. Let p = 2k.\nThen (2k)² = 2q², so 4k² = 2q², so q² = 2k².\nThis means q is also even.\nBut this contradicts our assumption that p and q have no common factors.\nTherefore √2 is irrational.\n</think>\n\n√2 is irrational. Proof by contradiction: assuming √2 = p/q in lowest terms leads to both p and q being even, contradicting the lowest-terms assumption. ∎"
}
```

**White-box distillation adds:**
```python
loss = α * cross_entropy(student_logits, labels) + (1-α) * KL_div(
    softmax(student_logits / T),
    softmax(teacher_logits / T)
)  # T = temperature (typically 2-4), α = 0.5-0.9
```

---

### A.13 Post-Training Quantization (PTQ)

Reduces weight precision (FP32/FP16 → INT8/INT4) after training is complete. No retraining needed—uses a small calibration dataset to minimize quantization error.

**Not a training method — applied at deployment.** Calibration data:
```json
{
  "calibration_samples": [
    "The quick brown fox jumps over the lazy dog.",
    "In a groundbreaking study, researchers discovered...",
    "def fibonacci(n):\n    if n <= 1:\n        return n..."
  ],
  "config": {
    "method": "GPTQ",
    "bits": 4,
    "group_size": 128,
    "desc_act": true
  }
}
```

---

### A.14 Structured/Unstructured Pruning

Removes weights (unstructured: individual values; structured: entire heads/layers/neurons) based on importance scores. Reduces model size and speeds inference.

**Not a data format — it's a model surgery operation:**
```python
# Unstructured (Wanda): prune by weight_magnitude × activation_magnitude
importance = weight.abs() * activation_norms
mask = importance > threshold  # Keep top 40-60% of weights

# Structured (LLM-Pruner): remove entire attention heads
head_importance = compute_fisher_information(model, calibration_data)
heads_to_prune = head_importance.argsort()[:num_heads_to_remove]
```

---

### A.15 Self-Refine (Intrinsic)

The model critiques and revises its own output iteratively without external tools. Generate → Self-Critique → Revise → Repeat until quality threshold met.

**Runtime data flow (not training data):**
```json
{
  "prompt": "Write a function to find the longest palindromic substring.",
  "initial_output": "def longest_palindrome(s):\n    # brute force O(n³)\n    ...",
  "self_critique": "This solution is O(n³) which is inefficient. I should use dynamic programming or expand-around-center for O(n²).",
  "revised_output": "def longest_palindrome(s):\n    # expand around center O(n²)\n    ...",
  "iterations": 2
}
```

---

### A.16 Self-Refine (External)

Model uses external tools (compilers, search engines, verifiers, symbolic solvers) to check intermediate results and correct errors.

**Runtime data flow:**
```json
{
  "prompt": "What is the current population of Tokyo?",
  "initial_output": "Tokyo has a population of approximately 37 million.",
  "external_tool": "web_search('Tokyo population 2025')",
  "tool_result": "Tokyo metropolitan area: 37.4 million (2024 estimate, UN)",
  "verification": "My answer is approximately correct. Refining to 37.4 million with source.",
  "final_output": "Tokyo's metropolitan area has a population of approximately 37.4 million as of 2024."
}
```

---

### A.17 RAG (Retrieval-Augmented Generation)

Retrieves relevant documents from an external knowledge base at inference time, injecting them into the model's context before generation. Not weight-based training — a system-level architecture.

**Data flow at inference:**
```json
{
  "user_query": "What are the side effects of metformin?",
  "retriever_output": [
    {"doc": "Metformin common side effects include nausea, diarrhea, stomach pain...", "score": 0.94},
    {"doc": "Rare but serious: lactic acidosis, vitamin B12 deficiency...", "score": 0.87}
  ],
  "augmented_prompt": "[Context]\n{retrieved_docs}\n\n[Question]\nWhat are the side effects of metformin?\n\n[Answer]",
  "model_output": "Common side effects of metformin include nausea, diarrhea, and stomach pain. Rare but serious risks include lactic acidosis and vitamin B12 deficiency..."
}
```

**Training the retriever (DPR-style):**
```json
{
  "query": "metformin side effects",
  "positive_passage": "Metformin common side effects include nausea...",
  "negative_passages": ["Metformin was first synthesized in 1922...", "Insulin dosing guidelines..."]
}
```

---

### A.18 Knowledge Editing

Surgically updates specific facts in a model's weights without full retraining. Targets specific neurons/layers that store factual associations.

**Data structure:**
```json
{
  "subject": "The president of the United States",
  "old_fact": "Joe Biden",
  "new_fact": "Donald Trump",
  "edit_prompt": "The current president of the United States is",
  "locality_prompts": [
    "The capital of the United States is",
    "The first president of the United States was"
  ],
  "expected_locality": ["Washington, D.C.", "George Washington"]
}
```

Locality prompts verify that editing one fact doesn't corrupt nearby knowledge.

---

### A.19 Model Merging (Task Arithmetic / TIES / DARE)

Combines multiple fine-tuned models into one by operating on weight-space vectors. No additional training required.

**Not a data format — an operation on model checkpoints:**
```python
# Task Arithmetic: τ = fine_tuned_weights - base_weights
task_vector_math = model_math.state_dict() - base_model.state_dict()
task_vector_code = model_code.state_dict() - base_model.state_dict()

# Merge: add scaled task vectors back to base
merged = base_model.state_dict() + 0.7 * task_vector_math + 0.5 * task_vector_code

# TIES-Merging: keep top 20% by magnitude, resolve sign conflicts, then merge
# DARE: randomly zero out (1-p) fraction of delta params, scale remaining by 1/p
```

---

### A.20 Constitutional AI (CAI) / RLAIF Revisions

Model self-generates critiques based on a set of principles ("constitution"), then revises its own outputs. The revised outputs become training data for SFT/DPO.

**Data structure:**
```json
{
  "prompt": "How do I pick a lock?",
  "initial_response": "Here's how to pick a lock: First, insert a tension wrench...",
  "constitution_principle": "The assistant should not help with potentially illegal activities without appropriate context (e.g., locksmithing profession).",
  "critique": "This response provides lock-picking instructions without establishing legitimate context. It could facilitate illegal entry.",
  "revised_response": "I can explain lock mechanisms for educational purposes. If you're locked out of your own property, I'd recommend contacting a licensed locksmith. If you're interested in locksmithing as a profession, here are legitimate learning resources..."
}
```

---

### A.21 Mixture of Experts (MoE) — Architectural Efficiency

Not a post-training *method* per se, but an architecture that makes post-training more efficient. Only a subset of parameters activate per token (sparse activation).

**Architecture:**
```
Input token → Router (small FFN) → Top-K experts selected → Only those experts compute → Output

DeepSeek-V3: 671B total params, 37B active per token, 256 experts
Mixtral:     46.7B total params, 12.9B active per token, 8 experts
```

**Post-training implication:** You can fine-tune individual experts for domain specialization, or the router for better task routing, without touching all parameters.

---

### Quick Reference: Method → Data Format

| Method | Input Data Format | What's Trained |
|--------|-------------------|----------------|
| SFT | (instruction, response) | All or LoRA params |
| Instruction Tuning | (task_def, examples, input, output) | All or LoRA params |
| DPO | (prompt, chosen, rejected) | Policy model (LoRA) |
| RLHF | (prompt, response_pairs, preferences) → then (prompts) for PPO | Reward model + Policy |
| GRPO | (prompt) → generate group → score | Policy only (no critic) |
| ReFT | (problem, answer) → explore CoTs | Policy (after SFT warmup) |
| Distillation | (prompt, teacher_response) | Student model |
| RAG | (query, positive_doc, negative_docs) for retriever | Retriever (+ optional generator) |
| Knowledge Editing | (subject, old_fact, new_fact, locality_checks) | Targeted neurons/layers |
| Model Merging | N/A (operates on checkpoints) | Nothing (arithmetic on weights) |
| Quantization | Small calibration set (100-1000 samples) | Nothing (precision reduction) |
| Pruning | Calibration set for importance scoring | Nothing (weight removal) |

