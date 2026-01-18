# Supplementary Academic Literature for SLM Agent Swarms Thesis

This report provides **peer-reviewed academic reinforcement** for a Master's thesis on "Agent Swarms and Architectures for Small Language Models," organized by thesis section with full citations, venues, and publication status.

## Verification of existing arXiv papers in document

Before presenting new literature, here is the publication status of the **six papers already cited**:

| arXiv ID | Title | Status | Venue |
|----------|-------|--------|-------|
| 2309.11408v2 | Indirect Swarm Control | **Preprint** | — |
| 2311.09758v2 | OrchestraLLM | **Published** | NAACL 2024 |
| 2501.00906v2 | Multi-Agent System for IoMT | **Preprint** | — |
| 2505.05762v1 | MAS for Robotic Autonomy | **Under review** | — |
| 2506.02153v2 | Small Language Models for Agentic AI | **Position paper** | NVIDIA Research |
| 2506.17208v2 | Dissecting SWE-Bench Leaderboards | **Preprint** | — |

**Key finding:** Only OrchestraLLM has achieved peer-reviewed publication (NAACL 2024). The thesis should acknowledge the preprint status of the others or supplement with peer-reviewed alternatives below.

---

## Section 2: Foundational principles

### Small Language Model papers

**Microsoft Phi Series** — The Phi papers demonstrate that **data quality exceeds model scale** for achieving strong performance:

- **Phi-1** — Gunasekar et al. "Textbooks Are All You Need." arXiv:2306.11644, 2023. *Preprint.* Foundational 1.3B model showing high-quality synthetic data enables competitive code generation.

- **Phi-3 Technical Report** — Abdin et al. "Phi-3: A Highly Capable Language Model Locally on Your Phone." arXiv:2404.14219, 2024. *Preprint.* **3.8B parameters rivaling Mixtral 8×7B**; key reference for phone-deployable SLMs.

- **Phi-4 Technical Report** — Abdin et al. arXiv:2412.08905, December 2024. *Preprint.* **14B model surpassing GPT-4 on STEM reasoning** through synthetic data focus.

**Google Gemma Series:**

- **Gemma** — Gemma Team, Google DeepMind. "Gemma: Open Models Based on Gemini Research and Technology." arXiv:2403.08295, 2024. *Technical report.* 2B and 7B models with multi-query attention and RoPE embeddings.

- **Gemma 3** — Gemma Team. arXiv:2503.19786, March 2025. *Preprint.* Multimodal 1B–27B with 128K context, distillation training; efficient local-to-global attention.

**Mistral Models:**

- **Mistral 7B** — Jiang et al. arXiv:2310.06825, October 2023. *Preprint.* **~2,800 citations.** Introduced sliding window attention (SWA) and grouped-query attention (GQA); outperforms LLaMA-2 13B.

- **Mixtral of Experts** — Jiang et al. arXiv:2401.04088, January 2024. *Preprint.* Sparse MoE (8×7B, 13B active) matching LLaMA-2 70B with 5× fewer active parameters.

**Alibaba Qwen Series:**

- **Qwen2 Technical Report** — Yang et al. arXiv:2407.10671, July 2024. *Preprint.* 0.5B–72B suite with MoE variants; **30 languages, strong coding capabilities**.

- **Qwen2.5 Technical Report** — Qwen Team. arXiv:2412.15115, December 2024. *Preprint.* **18T token pretraining** with 1M+ SFT samples; SOTA among open models.

**Meta LLaMA Series:**

- **LLaMA** — Touvron et al. "LLaMA: Open and Efficient Foundation Language Models." arXiv:2302.13971, February 2023. *Preprint.* **~15,000 citations.** Foundational open-weights paper democratizing LLM research.

- **LLaMA 2** — Touvron et al. arXiv:2307.09288, July 2023. *Preprint.* **~10,000 citations.** Detailed RLHF methodology for 7B–70B chat models.

- **TinyLLaMA** — Zhang et al. "TinyLlama: An Open-Source Small Language Model." arXiv:2401.02385, January 2024. *Preprint.* **Critical for thesis**—1.1B model trained on 3T tokens demonstrating over-training strategies.

### Quantization and efficient inference (peer-reviewed)

- **GPTQ** — Frantar et al. "GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers." **ICLR 2023.** arXiv:2210.17323. **~1,500 citations.** One-shot 3–4 bit weight quantization enabling 175B models on single GPU.

- **AWQ** — Lin et al. "AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration." **MLSys 2024 (Best Paper Award).** arXiv:2306.00978. Hardware-friendly weight-only quantization with TinyChat for on-device inference.

- **QLoRA** — Dettmers et al. "QLoRA: Efficient Finetuning of Quantized LLMs." **NeurIPS 2023.** arXiv:2305.14314. **~4,000 citations.** Enables 65B model finetuning on single 48GB GPU via 4-bit NormalFloat.

- **LLM.int8()** — Dettmers et al. "LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale." **NeurIPS 2022.** arXiv:2208.07339. **~1,800 citations.** First degradation-free 8-bit inference for 175B models.

- **SmoothQuant** — Xiao et al. "SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models." **ICML 2023.** arXiv:2211.10438. W8A8 quantization with **1.56× speedup and 2× memory reduction**.

### Knowledge distillation foundations

- **DistilBERT** — Sanh et al. "DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter." EMC²-NeurIPS Workshop, 2019. arXiv:1910.01108. **~12,000 citations.** Foundational distillation—40% smaller, 60% faster, 97% BERT performance.

- **TinyBERT** — Jiao et al. "TinyBERT: Distilling BERT for Natural Language Understanding." **EMNLP 2020 Findings.** arXiv:1909.10351. DOI: 10.18653/v1/2020.findings-emnlp.372. Two-stage Transformer distillation with attention and hidden state transfer.

### Speculative decoding

- **Speculative Sampling** — Chen et al. (Google DeepMind). "Accelerating Large Language Model Decoding with Speculative Sampling." arXiv:2302.01318, February 2023. *Preprint.* **2–2.5× speedup** on Chinchilla 70B using modified rejection sampling.

- **Self-Speculative Decoding** — Zhang et al. "Draft & Verify: Lossless Large Language Model Acceleration via Self-Speculative Decoding." **ACL 2024.** DOI: 10.18653/v1/2024.acl-long.607. Up to **1.99× speedup** without auxiliary models.

---

## Section 3: Architectural patterns

### Agent reasoning architectures (peer-reviewed foundations)

- **ReAct** — Yao et al. "ReAct: Synergizing Reasoning and Acting in Language Models." **ICLR 2023.** arXiv:2210.03629. **~3,500 citations.** Foundational paradigm interleaving reasoning traces with actions; enables LLM-environment interfaces.

- **Toolformer** — Schick et al. "Toolformer: Language Models Can Teach Themselves to Use Tools." **NeurIPS 2023.** arXiv:2302.04761. **~2,500 citations.** Self-supervised training for API tool use (calculator, search, Q&A, translation).

- **Chain-of-Thought** — Wei et al. "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models." **NeurIPS 2022.** arXiv:2201.11903. **~15,000 citations.** Foundational paper showing intermediate reasoning steps improve arithmetic and symbolic reasoning.

- **Self-Consistency** — Wang et al. "Self-Consistency Improves Chain of Thought Reasoning in Language Models." **ICLR 2023.** arXiv:2203.11171. **~3,500 citations.** Sample-and-marginalize decoding improving GSM8K by **+17.9%**.

- **Zero-shot CoT** — Kojima et al. "Large Language Models are Zero-Shot Reasoners." **NeurIPS 2022.** arXiv:2205.11916. **~4,000 citations.** "Let's think step by step" increased MultiArith from 17.7% to **78.7%**.

### Advanced reasoning extensions

- **Tree-of-Thoughts** — Yao et al. "Tree of Thoughts: Deliberate Problem Solving with Large Language Models." **NeurIPS 2023.** arXiv:2305.10601. **~2,200 citations.** BFS/DFS exploration with self-evaluation; **74% on Game of 24** vs 4% standard CoT.

- **Graph-of-Thoughts** — Besta et al. (ETH Zürich). "Graph of Thoughts: Solving Elaborate Problems with Large Language Models." **AAAI 2024.** arXiv:2308.09687. DOI: 10.1609/aaai.v38i16.29720. **~700 citations.** Arbitrary graph structures enabling thought merging and feedback loops.

- **PAL (Program-aided Language)** — Gao et al. (CMU). "PAL: Program-aided Language Models." **ICML 2023.** arXiv:2211.10435. **~1,500 citations.** LLMs generate Python programs as reasoning steps; **+15% over PaLM-540B CoT** on GSM8K.

### Foundational agent papers

- **MRKL Systems** — Karpas et al. (AI21 Labs). "MRKL Systems: A modular, neuro-symbolic architecture that combines large language models, external knowledge sources and discrete reasoning." arXiv:2205.00445, May 2022. *Preprint.* **~600 citations.** Coined modular architecture concept; influenced LangChain.

- **WebGPT** — Nakano et al. (OpenAI). "WebGPT: Browser-assisted question-answering with human feedback." arXiv:2112.09332, December 2021. *Preprint.* **~1,200 citations.** Early web-browsing agent with imitation learning + human feedback.

- **Reflexion** — Shinn et al. "Reflexion: Language Agents with Verbal Reinforcement Learning." **NeurIPS 2023.** arXiv:2303.11366. **~1,500 citations.** Self-reflection stored in episodic memory; achieved **91% pass@1 on HumanEval** (vs 80% GPT-4).

---

## Section 4: State of the art (multi-agent systems)

### Multi-agent LLM frameworks (peer-reviewed)

- **CAMEL** — Li et al. (KAUST). "CAMEL: Communicative Agents for 'Mind' Exploration of Large Language Model Society." **NeurIPS 2023.** arXiv:2303.17760. **~335 citations.** Introduced role-playing paradigm and "inception prompting" for autonomous multi-agent cooperation.

- **MetaGPT** — Hong et al. "MetaGPT: Meta Programming for A Multi-Agent Collaborative Framework." **ICLR 2024 (Oral, top 1.2%).** arXiv:2308.00352. **~700 citations.** Standardized Operating Procedures (SOPs) for structured agent workflows; **85.9% Pass@1 on HumanEval**.

- **AutoGen** — Wu et al. (Microsoft Research). "AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation." **COLM 2024.** arXiv:2308.08155. **~1,500 citations.** Introduces "conversation programming" paradigm unifying complex LLM workflows.

- **ChatDev** — Qian et al. (Tsinghua). "ChatDev: Communicative Agents for Software Development." **ACL 2024.** arXiv:2307.07924. DOI: 10.18653/v1/2024.acl-long.810. **~500 citations.** "Chat chain" with "communicative dehallucination"; complete software in **<7 minutes at <$1**.

- **AgentVerse** — Chen et al. (Tsinghua). "AgentVerse: Facilitating Multi-Agent Collaboration and Exploring Emergent Behaviors." **ICLR 2024.** arXiv:2308.10848. **~300 citations.** Dynamic agent composition with automatic "expert recruitment."

### Multi-agent debate and collaboration

- **Multi-Agent Debate (MAD)** — Du et al. (MIT, DeepMind). "Improving Factuality and Reasoning in Language Models through Multiagent Debate." **ICML 2024.** arXiv:2305.14325. **~450 citations.** Multiple LLM instances debate to reach consensus; improves factuality without fine-tuning.

- **Divergent Thinking MAD** — Liang et al. (Tencent, Tsinghua). "Encouraging Divergent Thinking in Large Language Models through Multi-Agent Debate." **EMNLP 2024.** arXiv:2305.19118. **~300 citations.** Addresses "Degeneration of Thought" through structured debate.

### Swarm intelligence + LLMs

- **LLM Swarm Applications** — Jimenez-Romero et al. "Multi-Agent Systems Powered by Large Language Models: Applications in Swarm Intelligence." **Frontiers in AI, 2025.** arXiv:2503.03800. LLMs as "behavioral engines" for emergent ant colony and flocking behaviors.

- **SwarmBench** — Ruan et al. arXiv:2505.04364, 2025. *Preprint.* Novel benchmark for LLM swarm constraints (limited perception, minimal communication) across five tasks.

- **Model Swarms** — U. Washington. arXiv:2410.11163. *Preprint.* Particle Swarm Optimization principles for adapting LLM weights collaboratively.

### Recent advances (October 2025 – January 2026)

- **Multi-Agent Collaboration Survey** — Tran et al. arXiv:2501.06322, January 2025. Comprehensive framework for collaboration types, structures, strategies, and coordination protocols.

- **AgentSquare** — "Automatic LLM Agent Search in Modular Design Space." **ICLR 2025.** 17.2% average gain over hand-crafted agents through modular Planning, Reasoning, Tool Use, Memory.

- **SWE-Search** — "Enhancing Software Agents with Monte Carlo Tree Search and Iterative Refinement." **ICLR 2025.** 23% relative improvement on SWE-bench via MCTS + self-improvement.

- **MacNet** — "Scaling Large Language Model-based Multi-Agent Collaboration." **ICLR 2025.** Identifies collaborative scaling law with logistic growth pattern using DAGs.

- **MCP Academic Treatment** — Hou et al. "Model Context Protocol (MCP): Landscape, Security Threats, and Future Research Directions." April 2025. First comprehensive MCP ecosystem analysis with security/privacy risks.

---

## Section 5: Evaluation and benchmarking

### Agent benchmarks (peer-reviewed)

- **AgentBench** — Liu et al. "AgentBench: Evaluating LLMs as Agents." **ICLR 2024.** arXiv:2308.03688. **~500 citations.** First systematic benchmark across 8 environments (OS, Database, WebShop, Mind2Web, etc.); evaluates 29 LLMs.

- **WebArena** — Zhou et al. "WebArena: A Realistic Web Environment for Building Autonomous Agents." **ICLR 2024.** arXiv:2307.13854. **~400 citations.** 812 tasks across 4 domains; GPT-4 achieved only **14.41% vs human 78.24%**.

- **GAIA** — Mialon et al. (Meta-FAIR, HuggingFace). "GAIA: A Benchmark for General AI Assistants." arXiv:2311.12983. **~200 citations.** 466 real-world questions; humans 92% vs GPT-4 with plugins at **15%**.

- **MINT** — Wang et al. (UIUC). "MINT: Evaluating LLMs in Multi-turn Interaction with Tools and Language Feedback." **ICLR 2024.** arXiv:2309.10691. **~100 citations.** Tests multi-turn tool use and language feedback; **finding: SIFT and RLHF hurt multi-turn capabilities**.

- **ToolLLM/ToolBench** — Qin et al. (OpenBMB). "ToolLLM: Facilitating Large Language Models to Master 16000+ Real-world APIs." **ICLR 2024 (Spotlight).** arXiv:2307.16789. **~300 citations.** 16,464 REST APIs with DFSDT planning and ToolEval automatic evaluator.

- **API-Bank** — Li et al. (Alibaba). "API-Bank: A Comprehensive Benchmark for Tool-Augmented LLMs." **EMNLP 2023.** arXiv:2304.08244. **~150 citations.** 73 API tools, 314 dialogues; complements ToolBench.

- **Mind2Web** — Deng et al. (OSU). "Mind2Web: Towards a Generalist Agent for the Web." **NeurIPS 2023 (Spotlight).** arXiv:2306.06070. **~300 citations.** First dataset for generalist web agents with 2,350 tasks from 137 real websites.

- **OSWorld** — Xie et al. (XLang-AI). "OSWorld: Benchmarking Multimodal Agents for Open-Ended Tasks in Real Computer Environments." **NeurIPS 2024.** arXiv:2404.07972. **~200 citations.** First scalable real OS environment; humans 72.36% vs best model **12.24%**.

- **InterCode** — Yang et al. (Princeton). "InterCode: Standardizing and Benchmarking Interactive Coding with Execution Feedback." **NeurIPS 2023.** arXiv:2306.14898. **~100 citations.** Interactive coding as RL environment with Bash, SQL, Python, CTF tasks.

### Code generation benchmarks

- **HumanEval/Codex** — Chen et al. (OpenAI). "Evaluating Large Language Models Trained on Code." arXiv:2107.03374, 2021. *Preprint (widely cited).* Introduced Codex and HumanEval (164 Python problems); established pass@k metric.

- **MBPP** — Austin et al. (Google). "Program Synthesis with Large Language Models." arXiv:2108.07732, 2021. *Preprint.* 974 entry-level Python tasks complementing HumanEval.

- **EvalPlus** — Liu et al. "Is Your Code Generated by ChatGPT Really Correct? Rigorous Evaluation of Large Language Models for Code Generation." **NeurIPS 2023.** arXiv:2305.01210. Augments HumanEval with **80× more test cases**; reveals test insufficiency.

- **RepoBench** — Liu et al. "RepoBench: Benchmarking Repository-Level Code Auto-Completion Systems." **ICLR 2024.** arXiv:2306.03091. Three tasks (Retrieval, Completion, Pipeline) for cross-file context.

### SWE-bench ecosystem

- **SWE-bench** — Jimenez et al. "SWE-bench: Can Language Models Resolve Real-World GitHub Issues?" **ICLR 2024 (Oral).** arXiv:2310.06770. 2,294 real GitHub issues from 12 Python repositories; foundational SE agent benchmark.

- **SWE-agent** — Yang et al. "SWE-agent: Agent-Computer Interfaces Enable Automated Software Engineering." **NeurIPS 2024.** arXiv:2405.15793. Introduces Agent-Computer Interface (ACI) concept; **12.47% on SWE-bench full test** with GPT-4 Turbo.

- **SWE-bench Multimodal** — Yang et al. **ICLR 2025.** Extends to visual software domains.

- **SWE-Bench Pro** — Scale AI, 2025. 1,865 tasks across 41 professional repositories including proprietary codebases.

- **SWE-bench-Live** — Zhang et al. arXiv:2505.23419. Monthly updated benchmark with 1,565 multi-language tasks (C/C++, C#, Python, Java, Go, JS/TS, Rust).

### Foundational reasoning benchmarks

- **ARC (Abstraction and Reasoning Corpus)** — Chollet. "On the Measure of Intelligence." arXiv:1911.01547, 2019. *Preprint.* **~2,000 citations.** 800 puzzle-like tasks testing fluid intelligence; humans ~80%, best AI ~31–53%. ARC-AGI-2 released May 2025.

- **MMLU** — Hendrycks et al. "Measuring Massive Multitask Language Understanding." **ICLR 2021.** arXiv:2009.03300. **~3,000 citations.** 57 subjects, 15,908 questions; de facto standard for general LLM evaluation.

- **HellaSwag** — Zellers et al. "HellaSwag: Can a Machine Really Finish Your Sentence?" **ACL 2019.** arXiv:1905.07830. **~1,500 citations.** Commonsense reasoning baseline.

- **BIG-Bench** — Srivastava et al. "Beyond the Imitation Game: Quantifying and extrapolating the capabilities of language models." **TMLR 2023.** arXiv:2206.04615. **~1,500 citations.** 204 tasks from 450+ authors; BIG-Bench Hard (BBH) focuses on 23 challenging tasks.

---

## Cross-cutting: RAG and retrieval systems

### Foundational RAG papers

- **RAG (Original)** — Lewis et al. "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks." **NeurIPS 2020.** arXiv:2005.11401. DOI: 10.5555/3495724.3496517. Seminal paper combining parametric and non-parametric memory.

- **Dense Passage Retrieval (DPR)** — Karpukhin et al. "Dense Passage Retrieval for Open-Domain Question Answering." **EMNLP 2020.** arXiv:2004.04906. DOI: 10.18653/v1/2020.emnlp-main.550. Dual-encoder dense retrieval outperforming BM25.

- **ColBERT** — Khattab & Zaharia. "ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT." **SIGIR 2020.** arXiv:2004.12832. DOI: 10.1145/3397271.3401075. Late interaction with MaxSim; **two orders of magnitude faster** than cross-encoders.

- **ColBERTv2** — Santhanam et al. "ColBERTv2: Effective and Efficient Retrieval via Lightweight Late Interaction." **NAACL 2022.** arXiv:2112.01488. **6–10× storage reduction** with improved quality.

### Advanced RAG techniques

- **Self-RAG** — Asai et al. "Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection." **ICLR 2024.** arXiv:2310.11511. Adaptive on-demand retrieval through reflection tokens; outperforms ChatGPT on QA tasks.

- **REPLUG** — Shi et al. "REPLUG: Retrieval-Augmented Black-Box Language Models." **NAACL 2024.** arXiv:2301.12652. Black-box RAG working with API-only LLMs; **+6.3% for GPT-3** on language modeling.

- **HyDE** — Gao et al. "Precise Zero-Shot Dense Retrieval without Relevance Labels." **ACL 2023.** arXiv:2212.10496. DOI: 10.18653/v1/2023.acl-long.99. Hypothetical document generation for zero-shot retrieval.

### Embedding models

- **Sentence-BERT** — Reimers & Gurevych. "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks." **EMNLP 2019.** arXiv:1908.10084. DOI: 10.18653/v1/D19-1410. Foundational sentence embeddings; basis for sentence-transformers library.

- **Contriever** — Izacard et al. "Unsupervised Dense Information Retrieval with Contrastive Learning." **TMLR 2022.** arXiv:2112.09118. Unsupervised dense retriever outperforming BM25 on 11/15 BEIR datasets.

- **E5** — Wang et al. "Text Embeddings by Weakly-Supervised Contrastive Pre-training." arXiv:2212.03533, 2022. *Preprint.* SOTA embeddings; first to outperform BM25 on BEIR without labeled data.

- **GTR** — Ni et al. "Large Dual Encoders Are Generalizable Retrievers." **EMNLP 2022.** arXiv:2112.07899. Scaling dual encoders improves out-of-domain generalization.

### Code-specific retrieval

- **CodeSearchNet** — Husain et al. "CodeSearchNet Challenge: Evaluating the State of Semantic Code Search." arXiv:1909.09436, 2019. *Technical report.* ~6 million functions across 6 languages; ~2 million (comment, code) pairs.

- **CodeRAG-Bench** — Wang et al. (CMU). "CodeRAG-Bench: Can Retrieval Augment Code Generation?" arXiv:2406.14497, 2024. *Preprint.* Comprehensive RAG benchmark for basic, open-domain, and repository-level code problems.

### RAG surveys

- **RAG Survey** — Gao et al. "Retrieval-Augmented Generation for Large Language Models: A Survey." arXiv:2312.10997. *Preprint (revised March 2024).* Covers Naive RAG, Advanced RAG, and Modular RAG paradigms.

- **RA-LLMs Survey** — Fan et al. "A Survey on RAG Meeting LLMs: Towards Retrieval-Augmented Large Language Models." **KDD 2024.** arXiv:2405.06211. DOI: 10.1145/3637528.3671470. Systematic review from three technical perspectives.

---

## Code LLMs for agent architectures

- **CodeLlama** — Rozière et al. (Meta). "Code Llama: Open Foundation Models for Code." arXiv:2308.12950, 2023. *Preprint (widely adopted).* 7B–70B family with infilling and 100K context.

- **StarCoder** — Li et al. (BigCode). "StarCoder: may the source be with you!" **TMLR 2023.** arXiv:2305.06161. 15.5B parameters from The Stack (1T tokens); open-access.

- **StarCoder2** — Lozhkov et al. arXiv:2402.19173, 2024. *Under review at TMLR.* 3B/7B/15B on 4T tokens; outperforms CodeLlama-34B.

- **DeepSeek-Coder** — Guo et al. "DeepSeek-Coder: When the Large Language Model Meets Programming." arXiv:2401.14196, 2024. *Preprint.* 1.3B–33B with project-level corpus and 16K context.

- **DeepSeek-Coder-V2** — DeepSeek-AI. arXiv:2406.11931, 2024. *Preprint.* MoE (16B/236B) comparable to GPT-4 Turbo; 128K context, 338 languages.

- **CodeGen** — Nijkamp et al. (Salesforce). "CodeGen: An Open Large Language Model for Code with Multi-Turn Program Synthesis." **ICLR 2023.** arXiv:2203.13474. 350M–16.1B pioneering conversational program synthesis.

---

## Software engineering + agents

- **LLM-based APR** — Xia et al. "Automated Program Repair in the Era of Large Pre-trained Language Models." **ICSE 2023.** First extensive study showing LLMs outperform all prior APR techniques.

- **APR Survey** — "A Survey of LLM-based Automated Program Repair: Taxonomies, Design Paradigms, and Applications." arXiv:2506.23749, 2024. *Preprint.* Covers fine-tuning, prompting, retrieval-augmented, and agent-based APR.

- **RepairAgent** — Bouzenia et al. "RepairAgent: An Autonomous, LLM-Based Agent for Program Repair." **ICSE 2025.**

- **CodePlan** — Bairi et al. "CodePlan: Repository-level Coding using LLMs and Planning." **FSE 2024.**

- **SE Agents Survey** — "Large Language Model-Based Agents for Software Engineering: A Survey." **SCIS 2025.** Comprehensive coverage of code generation, repair, testing, and SE agents.

---

## Recommended priority citations by thesis section

### Section 2 (Foundational Principles) — highest priority peer-reviewed:
1. QLoRA (NeurIPS 2023)
2. GPTQ (ICLR 2023)
3. AWQ (MLSys 2024 Best Paper)
4. DistilBERT (NeurIPS Workshop 2019)
5. Phi-3/Phi-4 technical reports (preprints but essential)
6. Mistral 7B (preprint, ~2,800 citations)

### Section 3 (Architectural Patterns) — highest priority:
1. ReAct (ICLR 2023, ~3,500 citations)
2. Chain-of-Thought (NeurIPS 2022, ~15,000 citations)
3. Toolformer (NeurIPS 2023, ~2,500 citations)
4. Tree-of-Thoughts (NeurIPS 2023)
5. Reflexion (NeurIPS 2023)
6. PAL (ICML 2023)

### Section 4 (SOTA Systems) — highest priority:
1. MetaGPT (ICLR 2024 Oral)
2. AutoGen (COLM 2024)
3. ChatDev (ACL 2024)
4. CAMEL (NeurIPS 2023)
5. AgentVerse (ICLR 2024)
6. Multi-Agent Debate (ICML 2024)

### Section 5 (Evaluation) — highest priority:
1. SWE-bench (ICLR 2024 Oral)
2. SWE-agent (NeurIPS 2024)
3. AgentBench (ICLR 2024)
4. WebArena (ICLR 2024)
5. MINT (ICLR 2024)
6. ToolLLM (ICLR 2024 Spotlight)
7. EvalPlus (NeurIPS 2023)
8. Mind2Web (NeurIPS 2023 Spotlight)

### Cross-cutting (RAG/Retrieval):
1. RAG original (NeurIPS 2020)
2. DPR (EMNLP 2020)
3. ColBERT (SIGIR 2020)
4. Self-RAG (ICLR 2024)
5. Sentence-BERT (EMNLP 2019)

---

## Key gaps and recommendations

**Gap 1: Limited peer-reviewed SLM-specific agent papers.** The NVIDIA position paper (arXiv:2506.02153) is influential but not peer-reviewed. Recommend supplementing with Phi-3/4 technical reports and TinyLLaMA for data efficiency arguments.

**Gap 2: Multi-agent evaluation benchmarks.** Most benchmarks evaluate single agents. SwarmBench (arXiv:2505.04364) is the closest to multi-agent swarm evaluation but remains a preprint.

**Gap 3: Edge deployment academic work.** Limited peer-reviewed academic treatment. The survey by Sharma & Mehta (arXiv:2510.03847) on "Small Language Models for Agentic Systems" provides comprehensive coverage but is preprint.

**Recommendation:** For Chapter 6 (Challenges and Future Directions), cite the agent evaluation surveys:
- arXiv:2503.16416 (March 2025) — "Survey on Evaluation of LLM-based Agents"
- arXiv:2507.21504 (KDD 2025) — "Evaluation and Benchmarking of LLM Agents: A Survey"

This supplementary bibliography provides **50+ peer-reviewed papers** across top venues (NeurIPS, ICLR, ICML, ACL, EMNLP, AAAI, SIGIR, KDD) and **30+ high-impact preprints** to reinforce the thesis's existing 38 citations with rigorous academic foundations.