# Model Post-Training Papers

---

## Post-Training, Alignment, and Steering

### 0. Public Fully Open Source

**Title:** OLMo 3: Open Language Model with 200T Tokens
- **Release:** December 2025
- **Authors:** Allyson Ettinger, Amanda Bertsch, et al. (Team OLMo)
- **ArXiv:** [arXiv:2512.13961](https://arxiv.org/abs/2512.13961)
- **Why it's worth reading:** This paper is the current gold standard for open-source transparency in LLM development. By releasing not just the weights, but the full training data, code, and intermediate checkpoints, Allen AI provides the most comprehensive "white-box" view into how a state-of-the-art model is built. It is essential for understanding data curation at massive scale and the nuances of training stability when pushing toward the 200-trillion-token frontier.

### 1. The Reasoning Model Roadmap

**Title:** Towards a Mechanistic Understanding of Large Reasoning Models: A Survey of Training, Inference, and Failures
- **Release:** January 2026
- **Authors:** Yi Hu, Jiaqi Gu, et al.
- **ArXiv:** [arXiv:2601.19928](https://arxiv.org/abs/2601.19928)
- **Significance:** This is the first major 2026 survey to explicitly categorize the post-training pipeline of "Reasoning Models" (like DeepSeek-R1), focusing on the internal circuitry that emerges during RL.

### 2. The Comprehensive Post-Training Survey

**Title:** A Survey on Post-training of Large Language Models
- **Release:** March 2025 (Original) | August 2025 (v3 Revision)
- **Authors:** Guiyao Tie, Zeli Zhao, et al.
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
- **Authors:** Jan Wehner, Sahar Abdelnabi, et al.
- **ArXiv:** [arXiv:2502.19649](https://arxiv.org/abs/2502.19649)
- **Significance:** Provides the definitive taxonomic framework for "Steering." It's essential for understanding how to operationalize internal model representations for control without retraining.

### 4. The RL-to-Steering Bridge

**Title:** Small Vectors, Big Effects: A Mechanistic Study of RL-Induced Reasoning via Steering Vectors
- **Release:** September 2025
- **Authors:** Viacheslav Sinii, Nikita Balagansky, et al.
- **ArXiv:** [arXiv:2509.06608](https://arxiv.org/abs/2509.06608)
- **Significance:** A critical mechanistic study proving that the reasoning capabilities gained through RL (like verifiable rewards) are represented as specific directions in the latent space that can be isolated and steered.

> **Note on Implementation:** If you are prioritizing the "Verifiable Rewards" (RLVR) aspect of your curriculum, the January 2026 Reasoning Model survey (2601.19928) contains the most current analysis on how RL stability is maintained when training models to generate their own internal reasoning chains.

---

## Deep Dive on Steering: Q1 2026

The first quarter of 2026 has been particularly prolific for "Steering" research, moving past simple additive vectors toward **unified theories** and **cross-modal** applications.

### 1. The "Theory" Breakthrough

**Title:** Why Steering Works: Toward a Unified View of Language Model Parameter Dynamics
- **Release:** February 2026
- **Authors:** Ziwen Xu, Chenyan Wu, et al.
- **ArXiv:** [arXiv:2602.02343](https://arxiv.org/abs/2602.02343)
- **Why it's worth reading:** This paper finally provides a unified mathematical framework for comparing SFT, LoRA, and Activation Steering. It introduces the **"Preference-Utility Manifold,"** explaining that steering works by shifting representations along a "target-concept direction" while trying to keep the model on the "valid-generation manifold." It's essential for understanding why steering often "breaks" model fluency at high intensities.

### 2. The "Bridge" to Weight Editing

**Title:** Steer2Edit: From Activation Steering to Component-Level Editing
- **Release:** February 2026
- **Authors:** Chung-En Sun, Ge Yan, et al.
- **ArXiv:** [arXiv:2602.09870](https://arxiv.org/abs/2602.09870)
- **Why it's worth reading:** If you find inference-time steering too compute-heavy or "brittle," this paper is the solution. It proposes a framework to translate transient steering vectors into **permanent, rank-1 weight updates** for specific attention heads or MLP neurons. It claims a 17% improvement in safety alignment over traditional steering.

### 3. Solving Hallucination & Faithfulness

**Title:** ContextFocus: Activation Steering for Contextual Faithfulness in Large Language Models
- **Release:** January 2026
- **Authors:** Nikhil Anand, Shwetha Somasundaram, et al.
- **ArXiv:** [arXiv:2601.04131](https://arxiv.org/abs/2601.04131)
- **Why it's worth reading:** A practical application of steering for RAG systems. It identifies the "internal knowledge" vs. "external context" conflict directions in the model's latent space and uses steering to force the model to prioritize the provided context over its pre-trained weights without any fine-tuning.

### 4. Steering in Audio-Language Models

**Title:** Nudging Hidden States: Training-Free Model Steering for Chain-of-Thought Reasoning in Large Audio-Language Models
- **Release:** March 2026
- **Authors:** Lok-Lam Ieong, Chia-Chien Chen, et al.
- **ArXiv:** [arXiv:2603.14636](https://arxiv.org/abs/2603.14636)
- **Why it's worth reading:** This is the first major paper to prove **cross-modal steering**. It demonstrates that steering vectors derived from simple *text* samples can effectively guide the reasoning process in *audio* models (LALMs), suggesting that reasoning "concepts" are represented identically across different input modalities.

### 5. Advanced Intervention Methods (ICLR 2026)

**Title:** Faithful Bi-Directional Model Steering via Distribution Matching and Distributed Interchange Interventions
- **Conference:** ICLR 2026 (Published Feb 2026)
- **Authors:** Yuntai Bao, Xuhong Zhang, et al.
- **ArXiv:** [arXiv:2602.05234](https://arxiv.org/abs/2602.05234)
- **Why it's worth reading:** It moves beyond simple vector addition. It uses "Distributed Interchange Interventions" (DII) and a distribution-matching objective to ensure that when you steer a model (e.g., to be "more helpful"), the output distribution remains natural and doesn't fall into the "off-distribution" artifacts common in earlier RepE methods.

---

## Agentic Systems: Q1 2026


 **Title:** *When Reasoning Meets Its Laws*
- **Framework Name:** **LoRe** (Laws of Reasoning)
- **Authors:** Junyu Zhang, Yifan Sun, Jingyan Shen, Liu Ziyin, Paul Pu Liang, and Huan Zhang.
- **Correct arXiv ID:** **[arXiv:2512.17901](https://arxiv.org/abs/2512.17901)** (Released December 19, 2025)
- **Why it's worth reading:** Following the success of DeepSeek-R1, this paper formalizes the "Scaling Laws" for reasoning. It introduces the **Compute Law**, which posits that reasoning performance scales linearly with "test-time compute" (how long the model "thinks" before answering). For someone building agentic systems, this is the theoretical backbone for determining when to use expensive "Chain-of-Thought" (CoT) versus rapid-fire inference.


**Title:** *GLM-5: From Vibe Coding to Agentic Engineering*
- **Authors:** 
- ** ArXiv:** [arXiv:2602.157632](https://arxiv.org/abs/2602.15763)
- **Why it's worth reading:** This paper moves past simple RAG and basic prompting into **Interleaved Reasoning**. It details an architecture where the model performs a hidden CoT loop *between* every tool call, allowing for "self-correction" during multi-step tasks (like coding or supply chain management). It also introduces "Retention-Based" reasoning, allowing agents to maintain a long-term "inner monologue" across thousands of tokens.

### Executive Summary: The Q1 2026 Shift

The common thread in these latest papers is a move away from **Knowledge Scaling** (adding more facts) toward **Compute Scaling** (adding more logic). These three papers represent the current industry consensus: the future of LLMs isn't just a bigger brain, but a brain that knows how to use "thinking time" to solve complex, multi-turn problems.


## Cognitive Density Retrospective: Q1 2026

The "Cognitive Density" retrospectives of March 2026 were primarily hosted across a few specialized technical blogs, preprint servers, and community forums that track the "Post-Training" transition. This movement—often called **"Logic Density over Parameter Scale"**—focuses on the efficiency of reasoning tokens versus raw model size.

Here are the primary links and sources where these discussions were held:

### The Industry Retrospectives (Blogs & Analysis)
* **Scalexa Tech Insights:** [**Cognitive Density: Why the "Reasoning" Era Favors Smaller Models**](https://scalexa.in/blogs.php)
    * *Focus:* This March 2026 series analyzes how "Synthetic Reasoning Traces" (the core of GPT-OSS and OLMo 3) allowed 20B models to bridge the gap with trillion-parameter giants. 
* **ArtificialAnalysis.ai:** [**Reasoning Benchmarks & Methodology (March 2026 Update)**](https://artificialanalysis.ai/methodology)
    * *Focus:* This is the definitive site for the "Intelligence Index v4.0" which popularized the **"Reasoning Tokens per Second"** metric that defined the March retrospectives.
* **Switas Strategy:** [**The Future of AGI: 5 Breakthroughs Defining April 2026**](https://www.switas.com/articles/the-future-of-agi-5-breakthroughs-defining-april-2026)
    * *Focus:* A follow-up to the March retrospectives, detailing the transition from "Generative AI" to "Autonomous Agentic AI" (specifically discussing GPT-5.4 and GLM-5).

### Theoretical & Research Foundations (Preprints)
* **Preprints.org:** [**Unifying Ontology Construction and Semantic Alignment... Logic Density over Parameter Scale**](https://www.preprints.org/manuscript/202603.1005)
    * *Focus:* Published in March 2026, this paper formalizes the "6D Probabilistic Wall" and argues for "dimensional ascension" through structured reasoning over statistical correlation.
* **medRxiv (Clinical Benchmarking):** [**Extracting Patient Reported Cannabis Use... A Benchmarking Study of GPT-OSS-20B**](https://www.medrxiv.org/content/10.64898/2026.03.06.26347824v1)
    * *Focus:* A March 9, 2026, real-world study that served as a "case study" in the retrospectives, proving that GPT-OSS-20B could outperform standard LLMs in high-precision clinical reasoning.

### Community & Peer Discussions
* **Hacker News (March 2026):** [**Discussion: Elon Musk pushes out more xAI founders as AI coding effort falters**](https://news.ycombinator.com/item?id=47366666)
    * *Focus:* A deep-dive discussion (see comments from "numbers_guy") regarding Grok 4.2 versus GPT-OSS 20B and why "Vibe Coding" was being replaced by "Agentic Engineering."
* **Reddit (r/LocalLLM):** [**"Why is GPT-OSS:20b so good, and is there anything that performs similarly?"**](https://www.reddit.com/r/LocalLLM/comments/1s9jo8v/why_is_gptoss20b_so_good_and_is_there_anything/)
    * *Focus:* A late-March discussion detailing the "Harmony" response format and why preserving CoT traces across turns is the secret sauce for 2026 agentic stability.

### Hints:
If you are diving into these, look for the **"MXFP4 (Microscaling)"** and **"Test-Time Compute"** sections. These are the two engineering breakthroughs that the retrospectives cite as the reason "Cognitive Density" finally became a viable alternative to massive server-side scaling. 
Checkout the  **ArtificialAnalysis.ai** report on **Amazon Nova Micro/Lite** (March 2026) particularly interesting, as it frames these models specifically within the "High-Speed Agentic" tier that these retrospectives advocate for. 

### Background:
**Title:** *GPT-OSS: Open-Weight Reasoning Models at Scale* 
* **Release:** August 5, 2025
* **Authors:** OpenAI (Reasoning & Alignment Team)
* **ArXiv:** [arXiv:2508.10925](https://arxiv.org/abs/2508.10925)
* **Why it's worth reading:** This is the foundational paper for **Cognitive Density**. It describes how OpenAI distilled the reasoning capabilities of its frontier "o-series" (and internal GPT-5 prototypes) into a 20B-parameter architecture. 
    * **Key Innovation:** It utilizes **MXFP4 (Microscaling)** quantization and "Synthetic Reasoning Traces" to allow a 20B model to match the logic of much larger proprietary systems.
    * **Hardware Impact:** It was the first "frontier-class" reasoner capable of running on a single consumer GPU (like an **RTX 4090** with 16GB–24GB VRAM), proving that high-level logic is a function of data quality and reasoning-specific training rather than raw parameter count.

