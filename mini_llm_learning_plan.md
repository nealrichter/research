# Self-Study AI Engineering Curriculum
### Architecture, Post-Training & Agents

---

## Module 1: Build — LLM Architecture & Pre-training

**Focus:** Understanding the "brain" of the LLM — Transformer architecture, scaling laws, and pre-training on massive datasets.

**Intro Course**
- [Generative AI for Everyone](https://www.deeplearning.ai/courses/generative-ai-for-everyone/) — *Andrew Ng*
  - ⏱ ~3 hrs | Introductory overview of how generative AI works, its capabilities, limitations, and business impact. Hands-on prompt engineering exercises included.


**Course 2**
- [Pretraining LLMs](https://www.deeplearning.ai/short-courses/pretraining-llms/) — *Sung Kim, Lucy Park*
  - ⏱ ~1 hr | Covers data preparation, model configuration, and cost considerations for pretraining from scratch or continuing pretraining on custom data.

**Course 3**
- [Build and Train an LLM with JAX](https://www.deeplearning.ai/short-courses/build-and-train-an-llm-with-jax/) — *Chris Achard* *(in partnership with Google)*
  - ⏱ ~1–2 hrs | Implement and train a 20M-parameter MiniGPT-style model from scratch using JAX and Flax/NNX — the library powering Google's Gemini.

**Course 4: Long Survey**
- [Generative AI with Large Language Models](https://learn.deeplearning.ai/courses/generative-ai-with-llms/information) — *AWS Team & DeepLearning.AI*
  - ⏱ ~16 hrs | Deep dive into the full LLM lifecycle: Transformer architecture, pre-training, fine-tuning, RLHF, scaling laws, and deployment. Intermediate Python required.
---

## Module 2: Post-Training

**Focus:** The post-training pipeline — transitioning from a raw base model to a helpful assistant using SFT, DPO, and RLHF.

**Course 1**
- [Post-training of LLMs](https://www.deeplearning.ai/short-courses/post-training-of-llms/) — *Banghua Zhu (UW / NexusFlow)*
  - ⏱ ~1–2 hrs | Covers SFT, DPO, and online RL concepts with guidance on curating high-quality training data for each method.

**Course 2**
- [Fine-Tuning and Reinforcement Learning for LLMs](https://www.deeplearning.ai/short-courses/finetuning-large-language-models/) — *Sharon Zhou (AMD)*
  - ⏱ ~1–2 hrs | 5-module course covering RLHF, PPO, GRPO, reward modeling, and LoRA for transforming pretrained models into reliable assistants.

**Course 3**
- [Reinforcement Learning from Human Feedback](https://www.deeplearning.ai/short-courses/reinforcement-learning-from-human-feedback/) — *Nikita Namjoshi*
  - ⏱ ~1–2 hrs | Focuses on RLHF as the primary alignment method — building reward models and applying RL to align LLMs with human values and preferences.

**Course 4**
- [Fine-tune an LLM for Function Calling](https://huggingface.co/learn/agents-course/bonus-unit1/introduction) — *Hugging Face*
  - ⏱ Self-paced module | Practical bonus unit on fine-tuning an open-source LLM specifically for structured tool/function-calling behavior.

---

## Module 3: Evaluating Agents

**Focus:** Observability, debugging, and evaluation of agent components. Covers testing examples, LLM-as-a-Judge, code-based evaluators, and structuring experiments to improve output quality.

**Course 1**
- [Evaluating AI Agents](https://www.deeplearning.ai/short-courses/evaluating-ai-agents/) — *John Gilhuly, Aman Khan*
  - ⏱ ~1–2 hrs | Learn to add observability to agents, set up evaluations with code-based and LLM-as-a-Judge approaches, and structure experiments to improve quality.
  - [GitHub: Evaluating AI Agents](https://github.com/ksm26/Evaluating-AI-Agents)

**Course 2**
- [Evaluating AI Agents](https://udemy.com/course/evaluating-ai-agents/) — *Yash Thakker*
  - ⏱ ~1 hr 6 min | Hands-on evaluation frameworks using LangSmith, Patronus, and PromptLayer. Covers quality, performance, and cost metrics plus A/B testing and production monitoring.

**Course 3**
- [Improving Accuracy of LLM Applications](https://www.deeplearning.ai/short-courses/improving-accuracy-of-llm-applications/) — *Sharon Zhou, Amit Sangani*
  - ⏱ ~1–2 hrs | Systematic approach to diagnosing and fixing inconsistent LLM outputs — covers evaluation pipelines, prompt iteration, and reliability techniques.

**Course 4**
- [Agent Observability and Evaluation](https://huggingface.co/learn/agents-course/bonus-unit2/introduction) — *Hugging Face*
  - ⏱ Self-paced module | Introduces observability tools (Langfuse, Arize), real-time monitoring dashboards, and live evaluation techniques including LLM-as-a-Judge and user feedback.

---

## Module 4: Agentic Systems Design

**Focus:** Giving the model hands and autonomy — Reflection, Tool Use, Planning, and Multi-Agent Orchestration using pure Python (no framework abstractions). *(Currently free)*

**Course**
- [Agentic AI](https://www.deeplearning.ai/short-courses/agentic-ai/) — *Andrew Ng*
  - ⏱ ~8 hrs | Teaches the four core agentic design patterns (Reflection, Tool Use, Planning, Multi-Agent Orchestration) built from first principles in Python before introducing frameworks.

---

## Module 5: Agents & App Building on Amazon Bedrock

**Course 1**
- [Serverless Agentic Workflows with Amazon Bedrock](https://www.deeplearning.ai/short-courses/serverless-agentic-workflows-with-amazon-bedrock/) — *Mike Chambers (AWS)*
  - ⏱ ~1 hr | Build and deploy a serverless agentic application with tools, code execution, and guardrails. Includes a hands-on customer service bot example.

**Course 2**
- [Serverless LLM Apps with Amazon Bedrock](https://www.deeplearning.ai/short-courses/serverless-llm-apps-amazon-bedrock/) — *Mike Chambers (AWS)*
  - ⏱ ~1 hr | Prompt and customize LLM responses on Bedrock. Build an event-driven audio summarizer using transcription + LLM in a serverless architecture.

**Course 3**
- [Amazon Bedrock: Generative AI, AI Agents, MCP, EVALs, RAG](https://udemy.com/course/mastering-aws-bedrock-build-intelligent-genai-applications/)
  - ⏱ ~8 hrs 14 min | Comprehensive Bedrock course covering Knowledge Bases with vector embeddings, Agents with Action Groups, Prompt Management, Guardrails, and open-source frameworks (CrewAI).

**Course 4**
- [Amazon Bedrock AgentCore: Build & Deploy any AI Agent on AWS](https://www.udemy.com/course/amazon-bedrock-agentcore-build-ai-agents-on-aws-hands-on/) — *Rahul Trisal*
  - ⏱ ~5 hrs 35 min | Hands-on deployment of agentic apps using AgentCore Runtime + Lambda + API Gateway. Covers observability (OpenTelemetry/CloudWatch), memory, identity, and gateway.

**Reference**
- [GitHub: Amazon Bedrock Samples](https://github.com/aws-samples/amazon-bedrock-samples)
