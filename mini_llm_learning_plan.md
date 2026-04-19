# LLM to Agents: A Self-Study AI Engineering Roadmap
### From Transformer Architecture & Pre-training → Post-Training & Alignment → Evaluation & Observability → Agentic Systems → Production Deployment on AWS

---

## Module 1: Build — LLM Architecture & Pre-training

**Focus:** Understanding the "brain" of the LLM — Transformer architecture, scaling laws, and pre-training on massive datasets.

**Intro Course**
- DeepLearning.AI [Generative AI for Everyone](https://www.deeplearning.ai/courses/generative-ai-for-everyone/) — *Andrew Ng*
  - ⏱ ~3 hrs | Introductory overview of how generative AI works, its capabilities, limitations, and business impact. Hands-on prompt engineering exercises included.
  - October 2023

**Course 1**
- DeepLearning.AI [Pretraining LLMs](https://www.deeplearning.ai/short-courses/pretraining-llms/) — *Sung Kim, Lucy Park*
  - ⏱ ~1 hr | Covers data preparation, model configuration, and cost considerations for pretraining from scratch or continuing pretraining on custom data.
  - July 2024

**Course 2**
- DeepLearning.AI [Build and Train an LLM with JAX](https://www.deeplearning.ai/short-courses/build-and-train-an-llm-with-jax/) — *Chris Achard* *(in partnership with Google)*
  - ⏱ ~1–2 hrs | Implement and train a 20M-parameter MiniGPT-style model from scratch using JAX and Flax/NNX — the library powering Google's Gemini.
  - March 2026

**Long Courses**

**Course 3**
- Coursera/DeepLearning.AI [Generative AI with Large Language Models](https://learn.deeplearning.ai/courses/generative-ai-with-llms/information) — *AWS Team & DeepLearning.AI*
  - ⏱ ~16 hrs | Deep dive into the full LLM lifecycle: Transformer architecture, pre-training, fine-tuning, RLHF, scaling laws, and deployment. Intermediate Python required.
  - October 2025

---

## Module 2: Post-Training

**Focus:** The post-training pipeline — transitioning from a raw base model to a helpful assistant using SFT, DPO, and RLHF.

**Course 1**
- DeepLearning.AI [Fine-tuning and Reinforcement Learning for LLMs: Intro to Post-Training](https://www.deeplearning.ai/courses/fine-tuning-and-reinforcement-learning-for-llms-intro-to-post-training/) — *Sharon Zhou (AMD)*
  - ⏱ ~1–2 hrs | 5-module course covering SFT, RLHF, PPO, GRPO, LoRA, evaluation/error analysis, synthetic data, and production pipelines. Built with AMD; methods are hardware-agnostic.
  - October 2025 *(older edition: [Finetuning Large Language Models](https://www.deeplearning.ai/short-courses/finetuning-large-language-models/), August 2023)*

**Course 2**
- DeepLearning.AI [Reinforcement Fine-Tuning LLMs with GRPO](https://learn.deeplearning.ai/courses/reinforcement-fine-tuning-llms-grpo/information) — *Travis Addair, Arnav Garg (Predibase)*
  - ⏱ ~1 hr 23 min | Technical deep dive into GRPO: reward function design, LLM-as-a-Judge, reward hacking, GRPO loss calculation, and fine-tuning for multi-step reasoning with minimal labeled data.
  - May 2025

> **TODO:** Find video link — *Hands-On GRPO Training: From GRPO Theory to Practice (2026)*
> Platform: Data Science Dojo / InnerWorkings.AI | Instructor: Chris McCormick | April 10, 2026

**Course 3**
- Hugging Face [Fine-tune an LLM for Function Calling](https://huggingface.co/learn/agents-course/bonus-unit1/introduction) — *Hugging Face*
  - ⏱ Self-paced module | Practical bonus unit on fine-tuning an open-source LLM specifically for structured tool/function-calling behavior.
  - [GitHub](https://github.com/huggingface/agents-course/blob/main/units/en/bonus-unit1/introduction.mdx) — February 2025

**Long Courses**

**Course 4**
- Udemy [LLM Mastery: Hands-on Code, Align and Master LLMs](https://www.udemy.com/course/llm-mastery-hands-on-code-align-and-master-llms/) — *Javier Ideami*
  - ⏱ ~18 hrs 44 min | Build an LLM from scratch in Python/PyTorch, code an alignment process (DPO-style), fine-tune with QLoRA, and explore attention mechanisms through advanced visualizations.
  - February 2026

---

## Module 3: Evaluating Agents

**Focus:** Observability, debugging, and evaluation of agent components. Covers testing examples, LLM-as-a-Judge, code-based evaluators, and structuring experiments to improve output quality.

**Course 1**
- DeepLearning.AI [Evaluating AI Agents](https://www.deeplearning.ai/short-courses/evaluating-ai-agents/) — *John Gilhuly, Aman Khan*
  - ⏱ ~1–2 hrs | Learn to add observability to agents, set up evaluations with code-based and LLM-as-a-Judge approaches, and structure experiments to improve quality.
  - [GitHub: Evaluating AI Agents](https://github.com/ksm26/Evaluating-AI-Agents)
  - February 2025

**Course 2** *(supplemental — tooling focus)*
- Udemy [Evaluating AI Agents](https://udemy.com/course/evaluating-ai-agents/) — *Yash Thakker*
  - ⏱ ~1 hr 6 min | Hands-on evaluation frameworks using LangSmith, Patronus, and PromptLayer. Covers quality, performance, and cost metrics plus A/B testing and production monitoring.
  - April 2025

**Course 3**
- DeepLearning.AI [Improving Accuracy of LLM Applications](https://www.deeplearning.ai/short-courses/improving-accuracy-of-llm-applications/) — *Sharon Zhou, Amit Sangani*
  - ⏱ ~1–2 hrs | Systematic approach to diagnosing and fixing inconsistent LLM outputs — covers evaluation pipelines, prompt iteration, and reliability techniques.
  - August 2024

**Course 4**
- Hugging Face [Agent Observability and Evaluation](https://huggingface.co/learn/agents-course/bonus-unit2/introduction) — *Hugging Face*
  - ⏱ Self-paced module | Introduces observability tools (Langfuse, Arize), real-time monitoring dashboards, and live evaluation techniques including LLM-as-a-Judge and user feedback.
  - [GitHub](https://github.com/huggingface/agents-course/blob/main/units/en/bonus-unit2/introduction.mdx) — March 2025

---

## Module 4: Agentic Systems Design

**Focus:** Giving the model hands and autonomy — Reflection, Tool Use, Planning, and Multi-Agent Orchestration. *(Currently free)*

**Long Courses**

**Course 1**
- DeepLearning.AI [Agentic AI](https://learn.deeplearning.ai/courses/agentic-ai/information) — *Andrew Ng*
  - ⏱ ~8 hrs | Teaches the four core agentic design patterns (Reflection, Tool Use, Planning, Multi-Agent Orchestration) built from first principles in Python before introducing frameworks.
  - October 2025

---

## Module 5: Agents & App Building on Amazon Bedrock

**Course 1**
- DeepLearning.AI [Serverless Agentic Workflows with Amazon Bedrock](https://www.deeplearning.ai/short-courses/serverless-agentic-workflows-with-amazon-bedrock/) — *Mike Chambers (AWS)*
  - ⏱ ~1 hr | Build and deploy a serverless agentic application with tools, code execution, and guardrails. Includes a hands-on customer service bot example.
  - October 2024

**Course 2**
- Udemy [Amazon Bedrock AgentCore: Build & Deploy any AI Agent on AWS](https://www.udemy.com/course/amazon-bedrock-agentcore-build-ai-agents-on-aws-hands-on/) — *Rahul Trisal*
  - ⏱ ~5 hrs 35 min | Hands-on deployment of agentic apps using AgentCore Runtime + Lambda + API Gateway. Covers observability (OpenTelemetry/CloudWatch), memory, identity, and gateway.
  - April 2026

**Course 3**
- Coursera/AWS [Amazon Bedrock Customization, Optimization & Automation](https://www.coursera.org/learn/amazon-bedrock-customization-optimization-automation) — *Russell Sayers, Alex G., Morgan Willis (AWS)*
  - ⏱ ~5 hrs | 2 modules:
    - **Module 1** — Fine-tuning, continued pre-training, model distillation, LangChain integration
    - **Module 2** — Bedrock Evaluation Jobs, prompt caching & routing, Bedrock Data Automation, Amazon Q Developer CLI
  - Part of the *AWS Generative AI and AI Agents with Amazon Bedrock* Professional Certificate | July 2025

**Long Courses**

**Course 4**
- Udemy [Amazon Bedrock: Generative AI, AI Agents, MCP, EVALs, RAG](https://udemy.com/course/mastering-aws-bedrock-build-intelligent-genai-applications/)
  - ⏱ ~8 hrs 14 min | Comprehensive Bedrock course covering Knowledge Bases with vector embeddings, Agents with Action Groups, Prompt Management, Guardrails, and open-source frameworks (CrewAI).
  - December 2025

**Course 5** *(supplemental — Python & TypeScript, embeddings/image models)*
- Udemy [Amazon Bedrock - The Complete Guide to AWS Generative AI](https://www.udemy.com/course/amazon-bedrock-aws-generative-ai/) — *Alex Dan, Bryan Krausen*
  - ⏱ ~8 hrs 3 min | End-to-end Bedrock course in Python and TypeScript. Covers text/image/embedding models, RAG with Knowledge Bases, LangChain integration, fine-tuning, and deploying APIs with Lambda + API Gateway.
  - April 2026

**Projects**
- Udemy [2026 Complete AWS Bedrock Generative AI Course + Projects](https://www.udemy.com/course/complete-aws-bedrock-generative-ai-course-projects) — *Patrik Szepesi*
  - ⏱ ~14 hrs 31 min | 6 real-world projects: meeting summarization, code generation, image generation, multi-agent system, voice AI agent (Nova Sonic), and web search agent with memory (Strands + DuckDuckGo). Covers AgentCore deployment, RFT, prompt management, batch inference, and short/long-term agent memory.
  - March 2026
- Coursera/Packt [Build Real-World End-to-End AI Agents using AWS Bedrock](https://www.coursera.org/learn/packt-build-real-world-end-to-end-ai-agents-using-aws-bedrock-bckaq) — *Packt*
  - ⏱ ~10 hrs | 10-module hands-on course: RAG, KnowledgeBases, Bedrock Agents, multi-agent collaboration, Lambda/DynamoDB integration, Redshift as structured data source, and Bedrock Flows for workflow orchestration.
  - February 2026

**Reference**
- [GitHub: Amazon Bedrock Samples](https://github.com/aws-samples/amazon-bedrock-samples)
- AWS Blog: [Reinforcement Fine-Tuning on Amazon Bedrock: Best Practices](https://aws.amazon.com/blogs/machine-learning/reinforcement-fine-tuning-on-amazon-bedrock-best-practices/) — *Nick McCarthy, Sapana Chaudhary, Shreyas Subramanian, Jennifer Zhu* | April 8, 2026
  - Covers RFT use cases (RLVR, RLAIF), dataset design, reward function strategy, hyperparameter tuning, and common pitfalls (reward hacking, instability).

---

## Module 6: MLOps

**Focus:** Operationalizing ML models — experiment tracking, model packaging, versioning, deployment, and production workflows.

**Course 1**
- Udemy [MLflow in Action - Master the art of MLOps using MLflow](https://www.udemy.com/course/mlflow-course/) — *J Garg - Real Time Learning*
  - ⏱ ~9 hrs 15 min | Covers MLOps fundamentals and all 4 MLflow components: Tracking, Model, Projects, and Registry. Includes end-to-end project deploying a model to AWS SageMaker with MLflow integration.
  - June 2025

**Course 2**
- Coursera/KodeKloud [Hands-On MLOps Fundamentals for ML Engineers Specialization](https://www.coursera.org/specializations/fundamentals-of-mlops) — *Mumshad Mannambeth*
  - ⏱ ~12 hrs total | 3-course specialization:
    - **Course 1** (5 hrs) — Data Engineering Essentials: Spark, Kafka, Airflow, Prefect, feature stores
    - **Course 2** (3 hrs) — ML Model Development & Tracking: MLflow experiment tracking, model registry, hyperparameter tuning
    - **Course 3** (4 hrs) — Deploy ML Models to Production: BentoML serving, CI/CD/CT, governance (GDPR/HIPAA), monitoring
  - Capstone: end-to-end insurance claim processing pipeline with MLflow + BentoML + Flask
  - March 2026

**Course 3**
- Udemy [MLOps Fundamentals: The Complete Guide to ML in Production](https://www.udemy.com/course/mlops-fundamentals-the-complete-guide-to-ml-in-production/) — *Data Universe, DCDG Partners*
  - ⏱ ~2 hrs 20 min | Conceptual overview of the full MLOps lifecycle: pipelines, CI/CD for ML, platform comparison (SageMaker, Vertex AI, Azure ML, MLflow, Kubeflow), monitoring, retraining, team roles, governance KPIs, and ROI. Includes real-world case studies (recommendations, fraud detection, churn).
  - April 2026

---

## Archive — Superseded / Overlapping Courses

*These courses are retained for reference but are largely covered by newer entries above.*

**Module 2 — Post-Training**
- DeepLearning.AI [Post-training of LLMs](https://www.deeplearning.ai/short-courses/post-training-of-llms/) — *Banghua Zhu (UW / NexusFlow)* | July 2025
  - Superseded by Module 2 Course 1 (Sharon Zhou, Oct 2025), which covers the same SFT/DPO/RL ground more comprehensively.
- DeepLearning.AI [Reinforcement Learning from Human Feedback](https://www.deeplearning.ai/short-courses/reinforcement-learning-from-human-feedback/) — *Nikita Namjoshi* | December 2023
  - RLHF now covered in Module 2 Course 1 and Course 2 (GRPO).

**Module 4 — Agentic Systems**
- DeepLearning.AI [Functions, Tools and Agents with LangChain](https://www.deeplearning.ai/short-courses/functions-tools-agents-langchain/) — *Harrison Chase (LangChain)* | October 2023
  - Tool use and agent patterns now covered more broadly in Module 4 Course 1 (Agentic AI). LangChain-specific LCEL syntax is dated.

**Module 5 — Amazon Bedrock**
- DeepLearning.AI [Serverless LLM Apps with Amazon Bedrock](https://www.deeplearning.ai/short-courses/serverless-llm-apps-amazon-bedrock/) — *Mike Chambers (AWS)* | February 2024
  - Superseded by Module 5 Course 1 (Serverless Agentic Workflows, Oct 2024), which covers the same serverless Bedrock pattern with agentic additions.
