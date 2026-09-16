# Early-Layer LLM Papers

A prioritized bibliography on the specialized behavior and practical use of the first 1–2 layers—and shallow layers generally—of transformer language models.

The tiers reflect **reading priority, not paper quality**:
- **Core Must-Reads** establish what early layers do and provide the main interpretive framework.
- **Strong but Goal-Dependent** papers become important for architecture, factuality, pruning, or inference work.
- **Specialized / Optional** papers are narrower applications or implementation techniques.

> **Repository note:** GitHub links below are author- or paper-linked repositories. MIT and Apache-2.0 repositories are conventionally open source. Entries marked CC BY-NC or “no license found” are public source code but have reuse restrictions or unclear reuse rights.

---

## Core Must-Reads

### 1. The Remarkable Robustness of LLMs: Stages of Inference?

**Focus:** Foundational robustness and stages of inference
- **Release:** June 2024 (NeurIPS 2025)
- **Authors:** Vedang Lad, Jin Hwa Lee, Wes Gurnee, Max Tegmark
- **Affiliation:** MIT
- **ArXiv:** [arXiv:2406.19384](https://arxiv.org/abs/2406.19384)
- **GitHub:** [vdlad/Remarkable-Robustness-of-LLMs](https://github.com/vdlad/Remarkable-Robustness-of-LLMs) — MIT license
- **Why it's worth reading:** The anchor paper for early-layer research. By deleting and swapping adjacent layers, the authors show that models retain 72–95% of top-1 accuracy without fine-tuning—but the *first* layer is a striking exception. They propose four broad stages of inference: **detokenization, feature engineering, prediction ensembling, and residual sharpening**. The first layer behaves like an extension of the embedding, combining subword information into context-sensitive representations; intervening there is much more damaging than removing many middle layers.

### 2. Inside the LLM Word Factory

**Focus:** The mechanics of detokenization at Layer 1
- **Release:** June 2026 (under review at EMNLP 2026)
- **Authors:** Benzi Busigin, Yuval Pinter
- **ArXiv:** [arXiv:2606.08562](https://arxiv.org/abs/2606.08562)
- **Why it's worth reading:** Pins down the mechanics behind the detokenization stage proposed by Lad et al. Activation-patching experiments localize English detokenization in Llama2-7B to a **two-stage process at Layer 1**: attention transmits a token-specific signal from nonfinal subwords, using sequential relays when needed, and the MLP composes that signal with the local embedding. The structure generalizes across 12 models from 8 families, but its depth depends on positional encoding—1–5 layers for RoPE models and 5–10 for learned-absolute models. An early-activation probe detects successful detokenization at 0.94–0.97 AUROC.

### 3. Attend First, Consolidate Later: On the Importance of Attention in Different LLM Layers

**Focus:** Early layers gather context; deeper layers consolidate it
- **Release:** September 2024 (BlackboxNLP 2024)
- **Authors:** Amit Ben-Artzy, Roy Schwartz
- **Affiliation:** Hebrew University of Jerusalem
- **ArXiv:** [arXiv:2409.03621](https://arxiv.org/abs/2409.03621)
- **Why it's worth reading:** Demonstrates a depth-dependent division of labor in decoder-only LLMs. Replacing previous-token hidden states with random vectors causes little damage when done in the final 30–50% of layers, but can reduce performance to chance when done in earlier layers. The results suggest that early layers depend heavily on attention to prior-token states, while later layers primarily consolidate information already gathered.

### 4. In-context Learning and Induction Heads

**Focus:** Why the canonical induction-head circuit needs two layers
- **Release:** September 2022
- **Authors:** Catherine Olsson, Nelson Elhage, Neel Nanda, Nicholas Joseph, Nova DasSarma, Tom Henighan, et al.
- **Affiliation:** Anthropic
- **ArXiv:** [arXiv:2209.11895](https://arxiv.org/abs/2209.11895)
- **Article:** [Transformer Circuits](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/)
- **Why it's worth reading:** A foundational mechanistic-interpretability study of the canonical induction-head circuit. One attention layer creates previous-token information; a later induction head uses it to find earlier `[A][B] … [A]` patterns and predict `[B]`. This composition requires at least two attention layers. The authors find that induction heads appear at the same training phase as a sharp increase in in-context learning and present evidence that these heads contribute substantially to general in-context learning.

### 5. Demystifying the Roles of LLM Layers in Retrieval, Knowledge, and Reasoning

**Focus:** Which layers support retrieval, knowledge, and reasoning
- **Release:** October 2025 (accepted at ICASSP 2026)
- **Authors:** Xinyuan Song, Keyu Wang, PengXiang Li, Lu Yin, Shiwei Liu
- **ArXiv:** [arXiv:2510.02091](https://arxiv.org/abs/2510.02091)
- **GitHub:** [Hik289/llm-layer-importance](https://github.com/Hik289/llm-layer-importance) — MIT license
- **Why it's worth reading:** Adds needed nuance to claims that deep layers are disposable. Under non-generative likelihood metrics, pruning many layers can preserve performance, with the initial layers being especially important. Generation-based evaluation instead reveals important middle- and deep-layer roles in reasoning and long-range coherence. Knowledge and retrieval are concentrated in shallower components, whereas reasoning accuracy relies more heavily on deeper layers.

### 6. The Curse of Depth in Large Language Models

**Focus:** Why deep Pre-LN layers become ineffective
- **Release:** February 2025 (NeurIPS 2025)
- **Authors:** Wenfang Sun, Xinyuan Song, Pengxiang Li, Lu Yin, Yefeng Zheng, Shiwei Liu
- **ArXiv:** [arXiv:2502.05795](https://arxiv.org/abs/2502.05795)
- **GitHub:** [lmsdss/LayerNorm-Scaling](https://github.com/lmsdss/LayerNorm-Scaling) — Apache-2.0 license
- **Why it's worth reading:** Goes beyond observing layer redundancy and proposes a mechanism for it. The paper argues that Pre-Layer Normalization causes residual-stream variance to grow with depth, pushing deep blocks toward identity-like behavior and reducing their contribution to training. Its **LayerNorm Scaling (LNS)** method scales normalized outputs inversely with the square root of depth, improving utilization of deeper layers and pretraining performance across models from 130M to 7B parameters.

---

## Strong but Goal-Dependent

### 7. ART: Attention Replacement Technique to Improve Factuality in LLMs

**Best for:** Factuality, hallucination mitigation, and attention intervention
- **Release:** April 2026
- **Authors:** Ziqin Luo, Yihao Quan, Xiaofeng Zhang, Xiaosong Yuan, Chen Shen
- **ArXiv:** [arXiv:2604.06393](https://arxiv.org/abs/2604.06393)
- **Why it's worth reading:** Introduces a training-free intervention for shallow-layer attention. The authors observe broad, near-uniform attention patterns in shallow layers and replace them with local attention patterns. This redirects attention toward relevant context and reduces hallucinations across multiple LLM architectures without fine-tuning or additional training data.

### 8. Mixture-of-Depths Attention

**Best for:** Architecture design and preserving shallow-layer signals
- **Release:** March 2026
- **Authors:** Lianghui Zhu, Yuxin Fang, Bencheng Liao, Shijie Wang, Tianheng Cheng, Zilong Huang, Chen Chen, Lai Wei, Yutao Zeng, Ya Wang, Yi Lin, Yu Li, Xinggang Wang
- **ArXiv:** [arXiv:2603.15619](https://arxiv.org/abs/2603.15619)
- **GitHub:** [hustvl/MoDA](https://github.com/hustvl/MoDA) — MIT license
- **Why it's worth reading:** Tackles **signal degradation**, where informative shallow-layer features are diluted by repeated residual updates. Mixture-of-Depths Attention (MoDA) lets each attention head use both current-layer sequence KV pairs and depth KV pairs from preceding layers. Its hardware-efficient implementation reaches 97.3% of FlashAttention-2 efficiency at a 64K sequence length. On 1.5B-parameter models, the paper reports a 0.2 average perplexity improvement and 2.11% higher average downstream performance for 3.7% additional FLOPs.

### 9. The Unreasonable Ineffectiveness of the Deeper Layers

**Best for:** Layer pruning and empirical depth redundancy
- **Release:** March 2024 (ICLR 2025)
- **Authors:** Andrey Gromov, Kushal Tirumala, Hassan Shapourian, Paolo Glorioso, Daniel A. Roberts
- **ArXiv:** [arXiv:2403.17887](https://arxiv.org/abs/2403.17887)
- **Conference:** [ICLR 2025 proceedings](https://proceedings.iclr.cc/paper_files/paper/2025/hash/cbabc2f70de2dd09f491a8715ec3e80f-Abstract-Conference.html)
- **GitHub:** [arcee-ai/PruneMe](https://github.com/arcee-ai/PruneMe) — unofficial public implementation; no license file found, so reuse rights are unclear
- **Why it's worth reading:** One of the strongest peer-reviewed demonstrations that common LLM evaluations can depend disproportionately on shallower layers. The authors identify a contiguous block to prune using layer similarity and then repair the model with lightweight QLoRA fine-tuning. On some open-weight models, removing up to half the layers causes little degradation on common question-answering benchmarks.

### 10. LayerSkip: Enabling Early Exit Inference and Self-Speculative Decoding

**Best for:** Early-exit inference and self-speculative decoding
- **Release:** April 2024 (ACL 2024)
- **Authors:** Mostafa Elhoushi, Akshat Shrivastava, Diana Liskovich, Basil Hosmer, Bram Wasti, Liangzhen Lai, et al.
- **ArXiv:** [arXiv:2404.16710](https://arxiv.org/abs/2404.16710)
- **DOI:** [10.18653/v1/2024.acl-long.681](https://doi.org/10.18653/v1/2024.acl-long.681)
- **GitHub:** [facebookresearch/LayerSkip](https://github.com/facebookresearch/LayerSkip) — CC BY-NC; public source with a noncommercial restriction
- **Why it's worth reading:** The main recognized precursor to newer early-exit systems such as WAVE. LayerSkip trains with progressively higher dropout rates in later layers and an early-exit loss shared across layers, then uses shallow-layer predictions as drafts that the remaining layers verify and correct. The paper reports speedups up to 2.16× on summarization, 1.82× on coding, and 2.0× on semantic parsing, without a separate draft model.

### 11. When Fewer Layers Break More Chains: Layer Pruning Harms Test-Time Scaling in LLMs

**Best for:** Understanding the limits and risks of layer pruning
- **Release:** October 2025 (revised July 2026)
- **Authors:** Keyu Wang, Tian Lyu, Guinan Su, Lu Yin, Marco Canini, Jonas Geiping, Shiwei Liu
- **ArXiv:** [arXiv:2510.22228](https://arxiv.org/abs/2510.22228)
- **GitHub:** [keyu-wang-2002/Layer-Pruning-Harms-Inference-Scaling](https://github.com/keyu-wang-2002/Layer-Pruning-Harms-Inference-Scaling) — official public code; no license file found, so reuse rights are unclear
- **Why it's worth reading:** An important counterweight to pruning results based on knowledge and short-answer benchmarks. The authors find that removing even one or two layers can severely damage long-chain reasoning and test-time scaling while leaving knowledge-intensive and shallow-reasoning scores relatively stable. Standard supervised fine-tuning does not restore the lost scaling behavior, warning that apparent layer redundancy is highly evaluation-dependent.

---

## Specialized / Optional

### 12. Shallow Focus, Deep Fixes: Enhancing Shallow Layers Vision Attention Sinks to Alleviate Hallucination in LVLMs

**Best for:** Multimodal models and vision hallucination
- **Release:** November 2025 (EMNLP 2025)
- **Authors:** Xiaofeng Zhang, Yihao Quan, Chen Shen, Chaochen Gu, Xiaosong Yuan, Shaotian Yan, Jiawei Cao, Hao Cheng, Kaijie Wu, Jieping Ye
- **ACL Anthology:** [2025.emnlp-main.174](https://aclanthology.org/2025.emnlp-main.174/)
- **DOI:** [10.18653/v1/2025.emnlp-main.174](https://doi.org/10.18653/v1/2025.emnlp-main.174)
- **GitHub:** [itsqyh/Shallow-Focus-Deep-Fixes](https://github.com/itsqyh/Shallow-Focus-Deep-Fixes) — MIT license
- **Why it's worth reading:** Connects LVLM hallucinations to layer-dependent vision attention sinks: shallow layers exhibit dense image-token sinks, while deeper layers exhibit sparser ones. The training-free **Enhancing Vision Attention Sinks (EVAS)** method identifies the densest visual-sink head in a shallow layer and broadcasts its attention matrix to other heads in that layer, strengthening attention to the image and reducing hallucination across multiple multimodal models.

### 13. WAVE: Window-Aware Vocabulary-Efficient Early-Exit for Training-Free LLM Acceleration

**Best for:** Training-free early exit without modifying model weights
- **Release:** July 2026 (ICML 2026)
- **Authors:** Seonggeun Kim, Gilha Lee, Hyun Kim
- **Conference:** [ICML 2026 paper page](https://icml.cc/virtual/2026/poster/62373) · [OpenReview](https://openreview.net/forum?id=hCY1J253Kf)
- **Why it's worth reading:** Addresses the overhead and shallow-layer collapse of earlier early-exit methods. WAVE calibrates an **exit window** that restricts exit decisions to a useful layer range, then constructs a lightweight vocabulary subset with a proxy LM head. The official ICML abstract reports an 87% reduction in per-layer exit overhead and up to 1.4× average speedup on Llama-2 7B while preserving output quality, without gradient-based training. It is also compatible with W4A16 quantization.

### 14. Transformer tricks: Precomputing the first layer

**Best for:** A compact, practical RoPE inference optimization
- **Release:** February 2024
- **Author:** Nils Graef
- **ArXiv:** [arXiv:2402.13388](https://arxiv.org/abs/2402.13388)
- **GitHub:** [OpenMachine-ai/transformer-tricks](https://github.com/OpenMachine-ai/transformer-tricks) — MIT license
- **Why it's worth reading:** A concise systems paper showing that much of the first layer can be precomputed for RoPE-based transformers such as LLaMA, Mistral, PaLM, and Gemma. This slightly lowers inference latency and cost per token. Because only one layer is optimized, relative savings shrink as model depth grows: the paper gives theoretical upper bounds of 25% for a four-layer model and 3% for a 32-layer model.
