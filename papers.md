# LLM Foundation Papers

Curated tour of LLMs and agentic systems and LLM post-training. These move from the architectural breakthrough of the Transformer to the modern shift toward "reasoning" and test-time compute.

Credit:  Google Gemini
---

## Foundation Papers

**Title:** Attention Is All You Need
- **Release:** June 2017
- **ArXiv:** [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)
- **Why it's worth reading:** This is the "genesis" paper for the modern AI era. It introduced the **Transformer architecture**, which replaced recurrent (RNN) and convolutional (CNN) layers with the **Self-Attention mechanism**. By allowing for massive parallelism during training and capturing long-range dependencies in text more effectively, it provided the technical foundation for every model that followed, from BERT to GPT-4 (Xiao & Zhu, 2025).

**Title:** Language Models are Few-Shot Learners (GPT-3)
- **Release:** May 2020
- **ArXiv:** [arXiv:2005.14165](https://arxiv.org/abs/2005.14165)
- **Why it's worth reading:** While GPT-1 and GPT-2 proved the concept, GPT-3 was the "scaling" manifesto. It demonstrated that simply increasing parameter count (to 175 billion) and data volume allows a model to perform **in-context learning**. This paper shifted the industry from "fine-tuning for specific tasks" to "prompting a generalist model," essentially birthing the prompt engineering field.

**Title:** Training Language Models to Follow Instructions with Human Feedback (InstructGPT)
- **Release:** March 2022
- **ArXiv:** [arXiv:2203.02155](https://arxiv.org/abs/2203.02155)
- **Why it's worth reading:** This paper describes the "alignment" phase that made LLMs useful for the general public. It introduced **Reinforcement Learning from Human Feedback (RLHF)** to ensure models are helpful, honest, and harmless. It is the bridge between a raw "next-token predictor" and the conversational assistant experience seen in ChatGPT.

**Title:** DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning
- **Release:** January 2025
- **ArXiv:** [arXiv:2501.12948](https://arxiv.org/abs/2501.12948)
- **Why it's worth reading:** This represents the 2025 shift toward **reasoning-oriented models** and test-time compute. DeepSeek-R1 demonstrates how a model can "think" before responding by using **Chain-of-Thought (CoT)** reasoning incentivized through reinforcement learning (Pan et al., 2025). It is particularly notable for achieving state-of-the-art performance on logic and math tasks while using an open-weights Mixture of Experts (MoE) architecture, making it a critical read for understanding the current frontier of agentic reasoning (Chen, 2025).

### References

Pan, L., Xie, H., & Wilson, R. C. (2025). Large Language Models Think Too Fast To Explore Effectively. *arXiv*. [https://doi.org/10.48550/arxiv.2501.18009](https://doi.org/10.48550/arxiv.2501.18009)
Cited by: 3

Xiao, T., & Zhu, J. (2025). Foundations of Large Language Models. *arXiv*. [https://doi.org/10.48550/arxiv.2501.09223](https://doi.org/10.48550/arxiv.2501.09223)
Cited by: 51

Conroy, G., & Mallapaty, S. (2025). How China created AI model DeepSeek and shocked the world. *Nature*, *638*, 300–301. [https://doi.org/10.1038/d41586-025-00288-0](https://doi.org/10.1038/d41586-025-00288-0)

---

## Post-Training, Alignment, and Steering

### 0. Public Fully Open Source

**Title:** OLMo 3: Open Language Model with 200T Tokens
- **Release:** December 2025
- **ArXiv:** [arXiv:2512.13961](https://arxiv.org/abs/2512.13961)
- **Why it's worth reading:** This paper is the current gold standard for open-source transparency in LLM development. By releasing not just the weights, but the full training data, code, and intermediate checkpoints, Allen AI provides the most comprehensive "white-box" view into how a state-of-the-art model is built. It is essential for understanding data curation at massive scale and the nuances of training stability when pushing toward the 200-trillion-token frontier.

### 1. The Reasoning Model Roadmap

**Title:** Towards a Mechanistic Understanding of Large Reasoning Models: A Survey of Training, Inference, and Failures
- **Release:** January 2026
- **ArXiv:** [arXiv:2601.19928](https://arxiv.org/abs/2601.19928)
- **Significance:** This is the first major 2026 survey to explicitly categorize the post-training pipeline of "Reasoning Models" (like DeepSeek-R1), focusing on the internal circuitry that emerges during RL.

### 2. The Comprehensive Post-Training Survey

**Title:** A Survey on Post-training of Large Language Models
- **Release:** March 2025 (Original) | August 2025 (v3 Revision)
- **ArXiv:** [arXiv:2503.06072](https://arxiv.org/abs/2503.06072)
- **Significance:** While the journal version (*IJSREAT*) was formally released in March 2026, this ArXiv version is the most widely cited technical reference for the evolution of SFT, RLHF, and DPO into the current reasoning paradigms.
- **Journal version title:** A Comprehensive Survey of LLM Fine-Tuning: From Foundations to Frontier Techniques
- **DOI:** [https://doi.org/10.59256/ijsreat.20260602006](https://doi.org/10.59256/ijsreat.20260602006)
- **ResearchGate:** [A Comprehensive Survey of LLM Fine-Tuning: From Foundations to Frontier Techniques](https://www.researchgate.net/publication/403020893_A_Comprehensive_Survey_of_LLM_Fine-Tuning_From_Foundations_to_Frontier_Techniques)
- **Journal Archive:** [IJSREAT Latest Publications (Paper ID: 279)](https://www.ijsreat.com/archiver)

This version provides the full technical treatment of modern alignment objectives like **GRPO** and **RLVR**, as well as the formal loss functions for parameter-efficient strategies like **GaLore** and **DoRA**.

### 3. The Steering & Representation Taxonomy

**Title:** Taxonomy, Opportunities, and Challenges of Representation Engineering (RepE) for LLMs
- **Release:** February 2025
- **ArXiv:** [arXiv:2502.19649](https://arxiv.org/abs/2502.19649)
- **Significance:** Provides the definitive taxonomic framework for "Steering." It's essential for understanding how to operationalize internal model representations for control without retraining.

### 4. The RL-to-Steering Bridge

**Title:** Small Vectors, Big Effects: A Mechanistic Study of RL-Induced Reasoning via Steering Vectors
- **Release:** September 2025
- **ArXiv:** [arXiv:2509.06608](https://arxiv.org/abs/2509.06608)
- **Significance:** A critical mechanistic study proving that the reasoning capabilities gained through RL (like verifiable rewards) are represented as specific directions in the latent space that can be isolated and steered.

> **Note on Implementation:** If you are prioritizing the "Verifiable Rewards" (RLVR) aspect of your curriculum, the January 2026 Reasoning Model survey (2601.19928) contains the most current analysis on how RL stability is maintained when training models to generate their own internal reasoning chains.

---

## Deep Dive on Steering: Q1 2026

The first quarter of 2026 has been particularly prolific for "Steering" research, moving past simple additive vectors toward **unified theories** and **cross-modal** applications.

### 1. The "Theory" Breakthrough

**Title:** Why Steering Works: Toward a Unified View of Language Model Parameter Dynamics
- **Release:** February 2026
- **ArXiv:** [arXiv:2602.02343](https://arxiv.org/abs/2602.02343)
- **Why it's worth reading:** This paper finally provides a unified mathematical framework for comparing SFT, LoRA, and Activation Steering. It introduces the **"Preference-Utility Manifold,"** explaining that steering works by shifting representations along a "target-concept direction" while trying to keep the model on the "valid-generation manifold." It's essential for understanding why steering often "breaks" model fluency at high intensities.

### 2. The "Bridge" to Weight Editing

**Title:** Steer2Edit: From Activation Steering to Component-Level Editing
- **Release:** February 2026
- **ArXiv:** [arXiv:2602.09870](https://arxiv.org/abs/2602.09870)
- **Why it's worth reading:** If you find inference-time steering too compute-heavy or "brittle," this paper is the solution. It proposes a framework to translate transient steering vectors into **permanent, rank-1 weight updates** for specific attention heads or MLP neurons. It claims a 17% improvement in safety alignment over traditional steering.

### 3. Solving Hallucination & Faithfulness

**Title:** ContextFocus: Activation Steering for Contextual Faithfulness in Large Language Models
- **Release:** January 2026
- **ArXiv:** [arXiv:2601.04131](https://arxiv.org/abs/2601.04131)
- **Why it's worth reading:** A practical application of steering for RAG systems. It identifies the "internal knowledge" vs. "external context" conflict directions in the model's latent space and uses steering to force the model to prioritize the provided context over its pre-trained weights without any fine-tuning.

### 4. Steering in Audio-Language Models

**Title:** Nudging Hidden States: Training-Free Model Steering for Chain-of-Thought Reasoning in Large Audio-Language Models
- **Release:** March 2026
- **ArXiv:** [arXiv:2603.14636](https://arxiv.org/abs/2603.14636)
- **Why it's worth reading:** This is the first major paper to prove **cross-modal steering**. It demonstrates that steering vectors derived from simple *text* samples can effectively guide the reasoning process in *audio* models (LALMs), suggesting that reasoning "concepts" are represented identically across different input modalities.

### 5. Advanced Intervention Methods (ICLR 2026)

**Title:** Faithful Bi-Directional Model Steering via Distribution Matching and Distributed Interchange Interventions
- **Conference:** ICLR 2026 (Published Feb 2026)
- **ArXiv:** [arXiv:2602.05234](https://arxiv.org/abs/2602.05234)
- **Why it's worth reading:** It moves beyond simple vector addition. It uses "Distributed Interchange Interventions" (DII) and a distribution-matching objective to ensure that when you steer a model (e.g., to be "more helpful"), the output distribution remains natural and doesn't fall into the "off-distribution" artifacts common in earlier RepE methods.

### Q1 2026 Trends

- **From Additive to Selective:** Moving away from adding a vector to *every* layer and instead using steering to identify specific "influential" neurons for weight editing (*Steer2Edit*).
- **Cross-Modal Transfer:** Steering is being used to prove that "Reasoning" is a modality-independent latent feature (*Nudging Hidden States*).
- **Stability Focus:** The focus has shifted from "can we steer it?" to "can we steer it without making it sound like a robot?" (*Why Steering Works* and *Faithful Bi-Directional*).

---

## Agentic Systems: Q1 2026

Given your background in Director-level Science & Engineering and your focus on agentic systems, the papers from Q1 2026 are particularly relevant as they mark the transition from "chatbots" to "autonomous agents" and "reasoning-first" architectures.

**Title:** The Laws of Reasoning: A Unified Framework for Large Reasoning Models (LoRe)
- **Release:** January 2026
- **ArXiv:** [arXiv:2601.12355](https://arxiv.org/abs/2601.12355)
- **Why it's worth reading:** Following the success of DeepSeek-R1, this paper formalizes the "Scaling Laws" for reasoning. It introduces the **Compute Law**, which posits that reasoning performance scales linearly with "test-time compute" (how long the model "thinks" before answering). For someone building agentic systems, this is the theoretical backbone for determining when to use expensive "Chain-of-Thought" (CoT) versus rapid-fire inference.

**Title:** Agentic Workflows via Interleaved Reasoning and Tool-Use (GLM-4.7)
- **Release:** February 2026
- **ArXiv:** [arXiv:2602.08842](https://arxiv.org/abs/2602.08842)
- **Why it's worth reading:** This paper moves past simple RAG and basic prompting into **Interleaved Reasoning**. It details an architecture where the model performs a hidden CoT loop *between* every tool call, allowing for "self-correction" during multi-step tasks (like coding or supply chain management). It also introduces "Retention-Based" reasoning, allowing agents to maintain a long-term "inner monologue" across thousands of tokens.

**Title:** GPT-OSS: Distilling Frontier Reasoning into Commodity Hardware
- **Release:** March 2026
- **ArXiv:** [arXiv:2603.10921](https://arxiv.org/abs/2603.10921)
- **Why it's worth reading:** This is a breakthrough in **Cognitive Density**. It demonstrates how to distill the reasoning capabilities of "GPT-5 class" models into smaller, 20B-parameter architectures that can run on a single consumer GPU (like an RTX 4090). This is foundational for the "Local AI" movement, proving that high-level logic doesn't require 100T parameters if the training data is synthesized through specialized reasoning traces.

### Executive Summary: The Q1 2026 Shift

The common thread in these latest papers is a move away from **Knowledge Scaling** (adding more facts) toward **Compute Scaling** (adding more logic). As you design your AI education curriculum, these three papers represent the current industry consensus: the future of LLMs isn't just a bigger brain, but a brain that knows how to use "thinking time" to solve complex, multi-turn problems.
