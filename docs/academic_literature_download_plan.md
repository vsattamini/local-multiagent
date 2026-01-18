# Academic Literature Download & Processing Plan
## Multi-Agent Software Engineering with Small Language Models

**Author:** Victor Sattamini  
**Date:** January 18, 2026  
**Purpose:** Systematic acquisition and organization of academic sources for Master's thesis

---

## Overview

This document provides a complete download and processing plan for **120+ academic papers** organized by thesis chapter. Each entry includes:
- Full citation
- Download link (arXiv, DOI, or venue)
- Publication status
- Processing priority
- Local file naming convention

---

## Directory Structure

```
thesis_literature/
├── ch2_foundations/
│   ├── slm_models/
│   ├── quantization/
│   ├── distillation/
│   └── inference/
├── ch3_architectures/
│   ├── reasoning_paradigms/
│   ├── agent_foundations/
│   └── tool_use/
├── ch4_sota/
│   ├── multi_agent_frameworks/
│   ├── software_engineering/
│   └── recent_advances/
├── ch5_evaluation/
│   ├── agent_benchmarks/
│   ├── code_benchmarks/
│   └── metrics/
├── cross_cutting/
│   ├── rag_retrieval/
│   ├── code_llms/
│   └── surveys/
└── metadata/
    ├── citations.bib
    ├── reading_notes/
    └── status_tracker.csv
```

---

## Chapter 2: Foundational Principles

### 2.1 Small Language Model Technical Reports

| # | Citation | arXiv/DOI | Status | Priority | Filename |
|---|----------|-----------|--------|----------|----------|
| 1 | Gunasekar et al. "Textbooks Are All You Need" (Phi-1) | 2306.11644 | Preprint | HIGH | `phi1_textbooks_2023.pdf` |
| 2 | Abdin et al. "Phi-3 Technical Report" | 2404.14219 | Tech Report | HIGH | `phi3_technical_2024.pdf` |
| 3 | Abdin et al. "Phi-4 Technical Report" | 2412.08905 | Tech Report | HIGH | `phi4_technical_2024.pdf` |
| 4 | Gemma Team "Gemma: Open Models" | 2403.08295 | Tech Report | HIGH | `gemma_technical_2024.pdf` |
| 5 | Gemma Team "Gemma 2 Technical Report" | 2408.00118 | Tech Report | HIGH | `gemma2_technical_2024.pdf` |
| 6 | Gemma Team "Gemma 3 Technical Report" | 2503.19786 | Tech Report | MEDIUM | `gemma3_technical_2025.pdf` |
| 7 | Jiang et al. "Mistral 7B" | 2310.06825 | Preprint | HIGH | `mistral7b_2023.pdf` |
| 8 | Jiang et al. "Mixtral of Experts" | 2401.04088 | Preprint | MEDIUM | `mixtral_moe_2024.pdf` |
| 9 | Yang et al. "Qwen2 Technical Report" | 2407.10671 | Tech Report | HIGH | `qwen2_technical_2024.pdf` |
| 10 | Qwen Team "Qwen2.5 Technical Report" | 2412.15115 | Tech Report | HIGH | `qwen2_5_technical_2024.pdf` |
| 11 | Touvron et al. "LLaMA" | 2302.13971 | Preprint | HIGH | `llama_original_2023.pdf` |
| 12 | Touvron et al. "LLaMA 2" | 2307.09288 | Preprint | HIGH | `llama2_2023.pdf` |
| 13 | Zhang et al. "TinyLlama" | 2401.02385 | Preprint | HIGH | `tinyllama_2024.pdf` |
| 14 | Bai et al. "Qwen2.5-Coder Technical Report" | 2409.12186 | Tech Report | HIGH | `qwen25_coder_2024.pdf` |

**Download commands:**
```bash
# Create directory
mkdir -p thesis_literature/ch2_foundations/slm_models

# Download all SLM papers
cd thesis_literature/ch2_foundations/slm_models

# Phi series
wget https://arxiv.org/pdf/2306.11644 -O phi1_textbooks_2023.pdf
wget https://arxiv.org/pdf/2404.14219 -O phi3_technical_2024.pdf
wget https://arxiv.org/pdf/2412.08905 -O phi4_technical_2024.pdf

# Gemma series
wget https://arxiv.org/pdf/2403.08295 -O gemma_technical_2024.pdf
wget https://arxiv.org/pdf/2408.00118 -O gemma2_technical_2024.pdf
wget https://arxiv.org/pdf/2503.19786 -O gemma3_technical_2025.pdf

# Mistral
wget https://arxiv.org/pdf/2310.06825 -O mistral7b_2023.pdf
wget https://arxiv.org/pdf/2401.04088 -O mixtral_moe_2024.pdf

# Qwen
wget https://arxiv.org/pdf/2407.10671 -O qwen2_technical_2024.pdf
wget https://arxiv.org/pdf/2412.15115 -O qwen2_5_technical_2024.pdf
wget https://arxiv.org/pdf/2409.12186 -O qwen25_coder_2024.pdf

# LLaMA
wget https://arxiv.org/pdf/2302.13971 -O llama_original_2023.pdf
wget https://arxiv.org/pdf/2307.09288 -O llama2_2023.pdf
wget https://arxiv.org/pdf/2401.02385 -O tinyllama_2024.pdf
```

### 2.2 Quantization & Efficient Inference

| # | Citation | arXiv/DOI | Venue | Status | Priority | Filename |
|---|----------|-----------|-------|--------|----------|----------|
| 15 | Frantar et al. "GPTQ" | 2210.17323 | ICLR 2023 | Peer-reviewed | HIGH | `gptq_iclr2023.pdf` |
| 16 | Lin et al. "AWQ" | 2306.00978 | MLSys 2024 | Peer-reviewed | HIGH | `awq_mlsys2024.pdf` |
| 17 | Dettmers et al. "QLoRA" | 2305.14314 | NeurIPS 2023 | Peer-reviewed | HIGH | `qlora_neurips2023.pdf` |
| 18 | Dettmers et al. "LLM.int8()" | 2208.07339 | NeurIPS 2022 | Peer-reviewed | HIGH | `llm_int8_neurips2022.pdf` |
| 19 | Xiao et al. "SmoothQuant" | 2211.10438 | ICML 2023 | Peer-reviewed | HIGH | `smoothquant_icml2023.pdf` |
| 20 | Frantar & Alistarh "SparseGPT" | 2301.00774 | ICML 2023 | Peer-reviewed | MEDIUM | `sparsegpt_icml2023.pdf` |
| 21 | Sheng et al. "FlexGen" | 2303.06865 | ICML 2023 | Peer-reviewed | MEDIUM | `flexgen_icml2023.pdf` |

**Download commands:**
```bash
mkdir -p thesis_literature/ch2_foundations/quantization
cd thesis_literature/ch2_foundations/quantization

wget https://arxiv.org/pdf/2210.17323 -O gptq_iclr2023.pdf
wget https://arxiv.org/pdf/2306.00978 -O awq_mlsys2024.pdf
wget https://arxiv.org/pdf/2305.14314 -O qlora_neurips2023.pdf
wget https://arxiv.org/pdf/2208.07339 -O llm_int8_neurips2022.pdf
wget https://arxiv.org/pdf/2211.10438 -O smoothquant_icml2023.pdf
wget https://arxiv.org/pdf/2301.00774 -O sparsegpt_icml2023.pdf
wget https://arxiv.org/pdf/2303.06865 -O flexgen_icml2023.pdf
```

### 2.3 Knowledge Distillation

| # | Citation | arXiv/DOI | Venue | Status | Priority | Filename |
|---|----------|-----------|-------|--------|----------|----------|
| 22 | Sanh et al. "DistilBERT" | 1910.01108 | NeurIPS WS 2019 | Peer-reviewed | HIGH | `distilbert_2019.pdf` |
| 23 | Jiao et al. "TinyBERT" | 1909.10351 | EMNLP 2020 | Peer-reviewed | HIGH | `tinybert_emnlp2020.pdf` |
| 24 | Sun et al. "MobileBERT" | 2004.02984 | ACL 2020 | Peer-reviewed | MEDIUM | `mobilebert_acl2020.pdf` |
| 25 | Hinton et al. "Distilling Knowledge" | 1503.02531 | NeurIPS WS 2015 | Peer-reviewed | HIGH | `hinton_distillation_2015.pdf` |

**Download commands:**
```bash
mkdir -p thesis_literature/ch2_foundations/distillation
cd thesis_literature/ch2_foundations/distillation

wget https://arxiv.org/pdf/1910.01108 -O distilbert_2019.pdf
wget https://arxiv.org/pdf/1909.10351 -O tinybert_emnlp2020.pdf
wget https://arxiv.org/pdf/2004.02984 -O mobilebert_acl2020.pdf
wget https://arxiv.org/pdf/1503.02531 -O hinton_distillation_2015.pdf
```

### 2.4 Speculative Decoding

| # | Citation | arXiv/DOI | Venue | Status | Priority | Filename |
|---|----------|-----------|-------|--------|----------|----------|
| 26 | Chen et al. "Speculative Sampling" | 2302.01318 | Preprint | HIGH | `speculative_sampling_2023.pdf` |
| 27 | Leviathan et al. "Fast Inference" | 2211.17192 | ICML 2023 | Peer-reviewed | HIGH | `fast_inference_icml2023.pdf` |
| 28 | Zhang et al. "Self-Speculative Decoding" | ACL 2024 | Peer-reviewed | MEDIUM | `self_speculative_acl2024.pdf` |

**Download commands:**
```bash
mkdir -p thesis_literature/ch2_foundations/inference
cd thesis_literature/ch2_foundations/inference

wget https://arxiv.org/pdf/2302.01318 -O speculative_sampling_2023.pdf
wget https://arxiv.org/pdf/2211.17192 -O fast_inference_icml2023.pdf
```

---

## Chapter 3: Architectural Patterns

### 3.1 Reasoning Paradigms

| # | Citation | arXiv/DOI | Venue | Status | Priority | Filename |
|---|----------|-----------|-------|--------|----------|----------|
| 29 | Wei et al. "Chain-of-Thought" | 2201.11903 | NeurIPS 2022 | Peer-reviewed | CRITICAL | `cot_neurips2022.pdf` |
| 30 | Wang et al. "Self-Consistency" | 2203.11171 | ICLR 2023 | Peer-reviewed | CRITICAL | `self_consistency_iclr2023.pdf` |
| 31 | Kojima et al. "Zero-shot CoT" | 2205.11916 | NeurIPS 2022 | Peer-reviewed | HIGH | `zero_shot_cot_neurips2022.pdf` |
| 32 | Yao et al. "Tree-of-Thoughts" | 2305.10601 | NeurIPS 2023 | Peer-reviewed | HIGH | `tot_neurips2023.pdf` |
| 33 | Besta et al. "Graph-of-Thoughts" | 2308.09687 | AAAI 2024 | Peer-reviewed | HIGH | `got_aaai2024.pdf` |
| 34 | Gao et al. "PAL" | 2211.10435 | ICML 2023 | Peer-reviewed | HIGH | `pal_icml2023.pdf` |
| 35 | Zhou et al. "Least-to-Most Prompting" | 2205.10625 | ICLR 2023 | Peer-reviewed | MEDIUM | `least_to_most_iclr2023.pdf` |

**Download commands:**
```bash
mkdir -p thesis_literature/ch3_architectures/reasoning_paradigms
cd thesis_literature/ch3_architectures/reasoning_paradigms

wget https://arxiv.org/pdf/2201.11903 -O cot_neurips2022.pdf
wget https://arxiv.org/pdf/2203.11171 -O self_consistency_iclr2023.pdf
wget https://arxiv.org/pdf/2205.11916 -O zero_shot_cot_neurips2022.pdf
wget https://arxiv.org/pdf/2305.10601 -O tot_neurips2023.pdf
wget https://arxiv.org/pdf/2308.09687 -O got_aaai2024.pdf
wget https://arxiv.org/pdf/2211.10435 -O pal_icml2023.pdf
wget https://arxiv.org/pdf/2205.10625 -O least_to_most_iclr2023.pdf
```

### 3.2 Agent Foundations

| # | Citation | arXiv/DOI | Venue | Status | Priority | Filename |
|---|----------|-----------|-------|--------|----------|----------|
| 36 | Yao et al. "ReAct" | 2210.03629 | ICLR 2023 | Peer-reviewed | CRITICAL | `react_iclr2023.pdf` |
| 37 | Schick et al. "Toolformer" | 2302.04761 | NeurIPS 2023 | Peer-reviewed | CRITICAL | `toolformer_neurips2023.pdf` |
| 38 | Shinn et al. "Reflexion" | 2303.11366 | NeurIPS 2023 | Peer-reviewed | HIGH | `reflexion_neurips2023.pdf` |
| 39 | Karpas et al. "MRKL Systems" | 2205.00445 | Preprint | HIGH | `mrkl_2022.pdf` |
| 40 | Nakano et al. "WebGPT" | 2112.09332 | Preprint | HIGH | `webgpt_2021.pdf` |
| 41 | Parisi et al. "TALM" | 2205.12255 | Preprint | MEDIUM | `talm_2022.pdf` |

**Download commands:**
```bash
mkdir -p thesis_literature/ch3_architectures/agent_foundations
cd thesis_literature/ch3_architectures/agent_foundations

wget https://arxiv.org/pdf/2210.03629 -O react_iclr2023.pdf
wget https://arxiv.org/pdf/2302.04761 -O toolformer_neurips2023.pdf
wget https://arxiv.org/pdf/2303.11366 -O reflexion_neurips2023.pdf
wget https://arxiv.org/pdf/2205.00445 -O mrkl_2022.pdf
wget https://arxiv.org/pdf/2112.09332 -O webgpt_2021.pdf
wget https://arxiv.org/pdf/2205.12255 -O talm_2022.pdf
```

### 3.3 Tool Use & Function Calling

| # | Citation | arXiv/DOI | Venue | Status | Priority | Filename |
|---|----------|-----------|-------|--------|----------|----------|
| 42 | Patil et al. "Gorilla" | 2305.15334 | Preprint | HIGH | `gorilla_2023.pdf` |
| 43 | Tang et al. "ToolAlpaca" | 2306.05301 | Preprint | MEDIUM | `toolalpaca_2023.pdf` |
| 44 | Hao et al. "ToolkenGPT" | 2305.11554 | NeurIPS 2023 | Peer-reviewed | HIGH | `toolkengpt_neurips2023.pdf` |

**Download commands:**
```bash
mkdir -p thesis_literature/ch3_architectures/tool_use
cd thesis_literature/ch3_architectures/tool_use

wget https://arxiv.org/pdf/2305.15334 -O gorilla_2023.pdf
wget https://arxiv.org/pdf/2306.05301 -O toolalpaca_2023.pdf
wget https://arxiv.org/pdf/2305.11554 -O toolkengpt_neurips2023.pdf
```

---

## Chapter 4: State of the Art

### 4.1 Multi-Agent Frameworks

| # | Citation | arXiv/DOI | Venue | Status | Priority | Filename |
|---|----------|-----------|-------|--------|----------|----------|
| 45 | Hong et al. "MetaGPT" | 2308.00352 | ICLR 2024 Oral | Peer-reviewed | CRITICAL | `metagpt_iclr2024.pdf` |
| 46 | Wu et al. "AutoGen" | 2308.08155 | COLM 2024 | Peer-reviewed | CRITICAL | `autogen_colm2024.pdf` |
| 47 | Qian et al. "ChatDev" | 2307.07924 | ACL 2024 | Peer-reviewed | CRITICAL | `chatdev_acl2024.pdf` |
| 48 | Li et al. "CAMEL" | 2303.17760 | NeurIPS 2023 | Peer-reviewed | HIGH | `camel_neurips2023.pdf` |
| 49 | Chen et al. "AgentVerse" | 2308.10848 | ICLR 2024 | Peer-reviewed | HIGH | `agentverse_iclr2024.pdf` |
| 50 | Du et al. "Multi-Agent Debate" | 2305.14325 | ICML 2024 | Peer-reviewed | HIGH | `mad_icml2024.pdf` |
| 51 | Liang et al. "Divergent Thinking MAD" | 2305.19118 | EMNLP 2024 | Peer-reviewed | MEDIUM | `divergent_mad_emnlp2024.pdf` |
| 52 | Park et al. "Generative Agents" | 2304.03442 | UIST 2023 | Peer-reviewed | HIGH | `generative_agents_uist2023.pdf` |

**Download commands:**
```bash
mkdir -p thesis_literature/ch4_sota/multi_agent_frameworks
cd thesis_literature/ch4_sota/multi_agent_frameworks

wget https://arxiv.org/pdf/2308.00352 -O metagpt_iclr2024.pdf
wget https://arxiv.org/pdf/2308.08155 -O autogen_colm2024.pdf
wget https://arxiv.org/pdf/2307.07924 -O chatdev_acl2024.pdf
wget https://arxiv.org/pdf/2303.17760 -O camel_neurips2023.pdf
wget https://arxiv.org/pdf/2308.10848 -O agentverse_iclr2024.pdf
wget https://arxiv.org/pdf/2305.14325 -O mad_icml2024.pdf
wget https://arxiv.org/pdf/2305.19118 -O divergent_mad_emnlp2024.pdf
wget https://arxiv.org/pdf/2304.03442 -O generative_agents_uist2023.pdf
```

### 4.2 Software Engineering Agents

| # | Citation | arXiv/DOI | Venue | Status | Priority | Filename |
|---|----------|-----------|-------|--------|----------|----------|
| 53 | Yang et al. "SWE-agent" | 2405.15793 | NeurIPS 2024 | Peer-reviewed | CRITICAL | `swe_agent_neurips2024.pdf` |
| 54 | Xia et al. "Agentless" | 2407.01489 | Preprint | HIGH | `agentless_2024.pdf` |
| 55 | Bouzenia et al. "RepairAgent" | ICSE 2025 | Peer-reviewed | HIGH | `repairagent_icse2025.pdf` |
| 56 | Bairi et al. "CodePlan" | FSE 2024 | Peer-reviewed | HIGH | `codeplan_fse2024.pdf` |
| 57 | Zhang et al. "AutoCodeRover" | 2404.05427 | Preprint | HIGH | `autocoderover_2024.pdf` |

**Download commands:**
```bash
mkdir -p thesis_literature/ch4_sota/software_engineering
cd thesis_literature/ch4_sota/software_engineering

wget https://arxiv.org/pdf/2405.15793 -O swe_agent_neurips2024.pdf
wget https://arxiv.org/pdf/2407.01489 -O agentless_2024.pdf
wget https://arxiv.org/pdf/2404.05427 -O autocoderover_2024.pdf
```

### 4.3 Recent Advances (Oct 2025 - Jan 2026)

| # | Citation | arXiv/DOI | Venue | Status | Priority | Filename |
|---|----------|-----------|-------|--------|----------|----------|
| 58 | Tran et al. "Multi-Agent Collaboration Survey" | 2501.06322 | Preprint | HIGH | `multiagent_survey_2025.pdf` |
| 59 | "AgentSquare" | ICLR 2025 | Peer-reviewed | HIGH | `agentsquare_iclr2025.pdf` |
| 60 | "SWE-Search" (MCTS) | ICLR 2025 | Peer-reviewed | HIGH | `swe_search_iclr2025.pdf` |
| 61 | "MacNet" (Collaborative Scaling) | ICLR 2025 | Peer-reviewed | HIGH | `macnet_iclr2025.pdf` |
| 62 | Hou et al. "MCP Landscape" | April 2025 | Preprint | MEDIUM | `mcp_landscape_2025.pdf` |
| 63 | Jimenez-Romero et al. "LLM Swarm Applications" | Frontiers AI | Peer-reviewed | HIGH | `llm_swarm_frontiers2025.pdf` |

**Download commands:**
```bash
mkdir -p thesis_literature/ch4_sota/recent_advances
cd thesis_literature/ch4_sota/recent_advances

wget https://arxiv.org/pdf/2501.06322 -O multiagent_survey_2025.pdf
# Note: ICLR 2025 papers may need OpenReview access
```

---

## Chapter 5: Evaluation & Benchmarking

### 5.1 Agent Benchmarks

| # | Citation | arXiv/DOI | Venue | Status | Priority | Filename |
|---|----------|-----------|-------|--------|----------|----------|
| 64 | Jimenez et al. "SWE-bench" | 2310.06770 | ICLR 2024 Oral | Peer-reviewed | CRITICAL | `swebench_iclr2024.pdf` |
| 65 | Liu et al. "AgentBench" | 2308.03688 | ICLR 2024 | Peer-reviewed | CRITICAL | `agentbench_iclr2024.pdf` |
| 66 | Zhou et al. "WebArena" | 2307.13854 | ICLR 2024 | Peer-reviewed | HIGH | `webarena_iclr2024.pdf` |
| 67 | Mialon et al. "GAIA" | 2311.12983 | Preprint | HIGH | `gaia_2023.pdf` |
| 68 | Wang et al. "MINT" | 2309.10691 | ICLR 2024 | Peer-reviewed | HIGH | `mint_iclr2024.pdf` |
| 69 | Qin et al. "ToolLLM" | 2307.16789 | ICLR 2024 Spotlight | Peer-reviewed | HIGH | `toollm_iclr2024.pdf` |
| 70 | Deng et al. "Mind2Web" | 2306.06070 | NeurIPS 2023 Spotlight | Peer-reviewed | HIGH | `mind2web_neurips2023.pdf` |
| 71 | Xie et al. "OSWorld" | 2404.07972 | NeurIPS 2024 | Peer-reviewed | HIGH | `osworld_neurips2024.pdf` |
| 72 | Yang et al. "InterCode" | 2306.14898 | NeurIPS 2023 | Peer-reviewed | MEDIUM | `intercode_neurips2023.pdf` |
| 73 | Li et al. "API-Bank" | 2304.08244 | EMNLP 2023 | Peer-reviewed | MEDIUM | `apibank_emnlp2023.pdf` |

**Download commands:**
```bash
mkdir -p thesis_literature/ch5_evaluation/agent_benchmarks
cd thesis_literature/ch5_evaluation/agent_benchmarks

wget https://arxiv.org/pdf/2310.06770 -O swebench_iclr2024.pdf
wget https://arxiv.org/pdf/2308.03688 -O agentbench_iclr2024.pdf
wget https://arxiv.org/pdf/2307.13854 -O webarena_iclr2024.pdf
wget https://arxiv.org/pdf/2311.12983 -O gaia_2023.pdf
wget https://arxiv.org/pdf/2309.10691 -O mint_iclr2024.pdf
wget https://arxiv.org/pdf/2307.16789 -O toollm_iclr2024.pdf
wget https://arxiv.org/pdf/2306.06070 -O mind2web_neurips2023.pdf
wget https://arxiv.org/pdf/2404.07972 -O osworld_neurips2024.pdf
wget https://arxiv.org/pdf/2306.14898 -O intercode_neurips2023.pdf
wget https://arxiv.org/pdf/2304.08244 -O apibank_emnlp2023.pdf
```

### 5.2 Code Generation Benchmarks

| # | Citation | arXiv/DOI | Venue | Status | Priority | Filename |
|---|----------|-----------|-------|--------|----------|----------|
| 74 | Chen et al. "Codex/HumanEval" | 2107.03374 | Preprint | CRITICAL | `codex_humaneval_2021.pdf` |
| 75 | Austin et al. "MBPP" | 2108.07732 | Preprint | HIGH | `mbpp_2021.pdf` |
| 76 | Liu et al. "EvalPlus" | 2305.01210 | NeurIPS 2023 | Peer-reviewed | HIGH | `evalplus_neurips2023.pdf` |
| 77 | Liu et al. "RepoBench" | 2306.03091 | ICLR 2024 | Peer-reviewed | HIGH | `repobench_iclr2024.pdf` |
| 78 | Zhuo et al. "BigCodeBench" | 2406.15877 | Preprint | MEDIUM | `bigcodebench_2024.pdf` |

**Download commands:**
```bash
mkdir -p thesis_literature/ch5_evaluation/code_benchmarks
cd thesis_literature/ch5_evaluation/code_benchmarks

wget https://arxiv.org/pdf/2107.03374 -O codex_humaneval_2021.pdf
wget https://arxiv.org/pdf/2108.07732 -O mbpp_2021.pdf
wget https://arxiv.org/pdf/2305.01210 -O evalplus_neurips2023.pdf
wget https://arxiv.org/pdf/2306.03091 -O repobench_iclr2024.pdf
wget https://arxiv.org/pdf/2406.15877 -O bigcodebench_2024.pdf
```

### 5.3 Foundational Benchmarks

| # | Citation | arXiv/DOI | Venue | Status | Priority | Filename |
|---|----------|-----------|-------|--------|----------|----------|
| 79 | Chollet "ARC" | 1911.01547 | Preprint | HIGH | `arc_chollet_2019.pdf` |
| 80 | Hendrycks et al. "MMLU" | 2009.03300 | ICLR 2021 | Peer-reviewed | HIGH | `mmlu_iclr2021.pdf` |
| 81 | Zellers et al. "HellaSwag" | 1905.07830 | ACL 2019 | Peer-reviewed | MEDIUM | `hellaswag_acl2019.pdf` |
| 82 | Srivastava et al. "BIG-Bench" | 2206.04615 | TMLR 2023 | Peer-reviewed | HIGH | `bigbench_tmlr2023.pdf` |

**Download commands:**
```bash
mkdir -p thesis_literature/ch5_evaluation/metrics
cd thesis_literature/ch5_evaluation/metrics

wget https://arxiv.org/pdf/1911.01547 -O arc_chollet_2019.pdf
wget https://arxiv.org/pdf/2009.03300 -O mmlu_iclr2021.pdf
wget https://arxiv.org/pdf/1905.07830 -O hellaswag_acl2019.pdf
wget https://arxiv.org/pdf/2206.04615 -O bigbench_tmlr2023.pdf
```

---

## Cross-Cutting: RAG & Retrieval

### Foundational RAG Papers

| # | Citation | arXiv/DOI | Venue | Status | Priority | Filename |
|---|----------|-----------|-------|--------|----------|----------|
| 83 | Lewis et al. "RAG" | 2005.11401 | NeurIPS 2020 | Peer-reviewed | CRITICAL | `rag_original_neurips2020.pdf` |
| 84 | Karpukhin et al. "DPR" | 2004.04906 | EMNLP 2020 | Peer-reviewed | CRITICAL | `dpr_emnlp2020.pdf` |
| 85 | Khattab & Zaharia "ColBERT" | 2004.12832 | SIGIR 2020 | Peer-reviewed | HIGH | `colbert_sigir2020.pdf` |
| 86 | Santhanam et al. "ColBERTv2" | 2112.01488 | NAACL 2022 | Peer-reviewed | HIGH | `colbertv2_naacl2022.pdf` |
| 87 | Asai et al. "Self-RAG" | 2310.11511 | ICLR 2024 | Peer-reviewed | HIGH | `selfrag_iclr2024.pdf` |
| 88 | Shi et al. "REPLUG" | 2301.12652 | NAACL 2024 | Peer-reviewed | MEDIUM | `replug_naacl2024.pdf` |
| 89 | Gao et al. "HyDE" | 2212.10496 | ACL 2023 | Peer-reviewed | MEDIUM | `hyde_acl2023.pdf` |

**Download commands:**
```bash
mkdir -p thesis_literature/cross_cutting/rag_retrieval
cd thesis_literature/cross_cutting/rag_retrieval

wget https://arxiv.org/pdf/2005.11401 -O rag_original_neurips2020.pdf
wget https://arxiv.org/pdf/2004.04906 -O dpr_emnlp2020.pdf
wget https://arxiv.org/pdf/2004.12832 -O colbert_sigir2020.pdf
wget https://arxiv.org/pdf/2112.01488 -O colbertv2_naacl2022.pdf
wget https://arxiv.org/pdf/2310.11511 -O selfrag_iclr2024.pdf
wget https://arxiv.org/pdf/2301.12652 -O replug_naacl2024.pdf
wget https://arxiv.org/pdf/2212.10496 -O hyde_acl2023.pdf
```

### Embedding Models

| # | Citation | arXiv/DOI | Venue | Status | Priority | Filename |
|---|----------|-----------|-------|--------|----------|----------|
| 90 | Reimers & Gurevych "Sentence-BERT" | 1908.10084 | EMNLP 2019 | Peer-reviewed | HIGH | `sbert_emnlp2019.pdf` |
| 91 | Izacard et al. "Contriever" | 2112.09118 | TMLR 2022 | Peer-reviewed | MEDIUM | `contriever_tmlr2022.pdf` |
| 92 | Wang et al. "E5" | 2212.03533 | Preprint | HIGH | `e5_embeddings_2022.pdf` |

**Download commands:**
```bash
wget https://arxiv.org/pdf/1908.10084 -O sbert_emnlp2019.pdf
wget https://arxiv.org/pdf/2112.09118 -O contriever_tmlr2022.pdf
wget https://arxiv.org/pdf/2212.03533 -O e5_embeddings_2022.pdf
```

---

## Cross-Cutting: Code LLMs

| # | Citation | arXiv/DOI | Venue | Status | Priority | Filename |
|---|----------|-----------|-------|--------|----------|----------|
| 93 | Rozière et al. "Code Llama" | 2308.12950 | Preprint | HIGH | `codellama_2023.pdf` |
| 94 | Li et al. "StarCoder" | 2305.06161 | TMLR 2023 | Peer-reviewed | HIGH | `starcoder_tmlr2023.pdf` |
| 95 | Lozhkov et al. "StarCoder2" | 2402.19173 | Under review | HIGH | `starcoder2_2024.pdf` |
| 96 | Guo et al. "DeepSeek-Coder" | 2401.14196 | Preprint | HIGH | `deepseek_coder_2024.pdf` |
| 97 | DeepSeek-AI "DeepSeek-Coder-V2" | 2406.11931 | Preprint | HIGH | `deepseek_coder_v2_2024.pdf` |
| 98 | Nijkamp et al. "CodeGen" | 2203.13474 | ICLR 2023 | Peer-reviewed | MEDIUM | `codegen_iclr2023.pdf` |

**Download commands:**
```bash
mkdir -p thesis_literature/cross_cutting/code_llms
cd thesis_literature/cross_cutting/code_llms

wget https://arxiv.org/pdf/2308.12950 -O codellama_2023.pdf
wget https://arxiv.org/pdf/2305.06161 -O starcoder_tmlr2023.pdf
wget https://arxiv.org/pdf/2402.19173 -O starcoder2_2024.pdf
wget https://arxiv.org/pdf/2401.14196 -O deepseek_coder_2024.pdf
wget https://arxiv.org/pdf/2406.11931 -O deepseek_coder_v2_2024.pdf
wget https://arxiv.org/pdf/2203.13474 -O codegen_iclr2023.pdf
```

---

## Cross-Cutting: Surveys

| # | Citation | arXiv/DOI | Venue | Status | Priority | Filename |
|---|----------|-----------|-------|--------|----------|----------|
| 99 | Gao et al. "RAG Survey" | 2312.10997 | Preprint | HIGH | `rag_survey_2024.pdf` |
| 100 | Fan et al. "RA-LLMs Survey" | 2405.06211 | KDD 2024 | Peer-reviewed | HIGH | `ra_llms_kdd2024.pdf` |
| 101 | Wang et al. "LLM Agents Survey" | 2309.07864 | Preprint | HIGH | `llm_agents_survey_2023.pdf` |
| 102 | Xi et al. "Rise of LLM Agents" | 2309.07864 | Preprint | HIGH | `rise_llm_agents_2023.pdf` |
| 103 | "SE Agents Survey" | SCIS 2025 | Peer-reviewed | HIGH | `se_agents_survey_scis2025.pdf` |

**Download commands:**
```bash
mkdir -p thesis_literature/cross_cutting/surveys
cd thesis_literature/cross_cutting/surveys

wget https://arxiv.org/pdf/2312.10997 -O rag_survey_2024.pdf
wget https://arxiv.org/pdf/2405.06211 -O ra_llms_kdd2024.pdf
wget https://arxiv.org/pdf/2309.07864 -O llm_agents_survey_2023.pdf
```

---

## Processing Pipeline

### Step 1: Batch Download Script

```bash
#!/bin/bash
# download_all_papers.sh

BASE_DIR="thesis_literature"
LOG_FILE="$BASE_DIR/download_log.txt"

echo "Starting paper download: $(date)" > $LOG_FILE

# Function to download with retry
download_paper() {
    url=$1
    output=$2
    max_retries=3
    retry_count=0
    
    while [ $retry_count -lt $max_retries ]; do
        wget -q --show-progress "$url" -O "$output"
        if [ $? -eq 0 ]; then
            echo "SUCCESS: $output" >> $LOG_FILE
            return 0
        fi
        retry_count=$((retry_count + 1))
        sleep 2
    done
    
    echo "FAILED: $output after $max_retries attempts" >> $LOG_FILE
    return 1
}

# Create all directories
mkdir -p $BASE_DIR/{ch2_foundations/{slm_models,quantization,distillation,inference},ch3_architectures/{reasoning_paradigms,agent_foundations,tool_use},ch4_sota/{multi_agent_frameworks,software_engineering,recent_advances},ch5_evaluation/{agent_benchmarks,code_benchmarks,metrics},cross_cutting/{rag_retrieval,code_llms,surveys},metadata/reading_notes}

# Download papers (abbreviated - full list in script)
echo "Downloading Chapter 2 papers..."
download_paper "https://arxiv.org/pdf/2306.11644" "$BASE_DIR/ch2_foundations/slm_models/phi1_textbooks_2023.pdf"
download_paper "https://arxiv.org/pdf/2404.14219" "$BASE_DIR/ch2_foundations/slm_models/phi3_technical_2024.pdf"
# ... continue for all papers

echo "Download complete: $(date)" >> $LOG_FILE
echo "Check $LOG_FILE for status"
```

### Step 2: BibTeX Generation

```bibtex
% thesis_literature/metadata/citations.bib

@article{wei2022chain,
  title={Chain-of-thought prompting elicits reasoning in large language models},
  author={Wei, Jason and Wang, Xuezhi and Schuurmans, Dale and others},
  journal={Advances in Neural Information Processing Systems},
  volume={35},
  pages={24824--24837},
  year={2022}
}

@inproceedings{yao2023react,
  title={ReAct: Synergizing reasoning and acting in language models},
  author={Yao, Shunyu and Zhao, Jeffrey and Yu, Dian and others},
  booktitle={International Conference on Learning Representations},
  year={2023}
}

@inproceedings{hong2024metagpt,
  title={MetaGPT: Meta Programming for A Multi-Agent Collaborative Framework},
  author={Hong, Sirui and Zhuge, Mingchen and Chen, Jonathan and others},
  booktitle={International Conference on Learning Representations},
  year={2024}
}

@inproceedings{jimenez2024swebench,
  title={SWE-bench: Can Language Models Resolve Real-World GitHub Issues?},
  author={Jimenez, Carlos E and Yang, John and Wettig, Alexander and others},
  booktitle={International Conference on Learning Representations},
  year={2024}
}

% ... continue for all papers
```

### Step 3: Reading Status Tracker

```csv
# thesis_literature/metadata/status_tracker.csv
paper_id,filename,status,priority,chapter,notes_taken,cited_in_thesis,date_read
1,phi1_textbooks_2023.pdf,downloaded,HIGH,ch2,no,no,
2,phi3_technical_2024.pdf,downloaded,HIGH,ch2,no,no,
3,gptq_iclr2023.pdf,downloaded,HIGH,ch2,no,no,
29,cot_neurips2022.pdf,read,CRITICAL,ch3,yes,yes,2026-01-15
36,react_iclr2023.pdf,read,CRITICAL,ch3,yes,yes,2026-01-16
```

### Step 4: Reading Notes Template

```markdown
# Reading Notes: [Paper Title]

**Citation:** [Full citation]
**Date Read:** [Date]
**Relevance:** [Chapter/Section]

## Summary (3-5 sentences)


## Key Contributions
1. 
2. 
3. 

## Methods


## Results


## Relevance to Thesis
- How does this connect to RQ1/RQ2/RQ3/RQ4?
- What claims can I support with this paper?
- Are there limitations I should acknowledge?

## Quotes to Cite
- "..." (p. X)

## Related Papers
- [List papers this cites or that cite this]

## Questions/Follow-ups
- 
```

---

## Priority Processing Order

### Week 1: Critical Foundation Papers (15 papers)
1. Chain-of-Thought (Wei et al.)
2. ReAct (Yao et al.)
3. Toolformer (Schick et al.)
4. SWE-bench (Jimenez et al.)
5. MetaGPT (Hong et al.)
6. AutoGen (Wu et al.)
7. ChatDev (Qian et al.)
8. GPTQ (Frantar et al.)
9. QLoRA (Dettmers et al.)
10. RAG Original (Lewis et al.)
11. AgentBench (Liu et al.)
12. Codex/HumanEval (Chen et al.)
13. Phi-3 Technical Report
14. Qwen2.5-Coder Technical Report
15. SWE-agent (Yang et al.)

### Week 2: High Priority Papers (25 papers)
- All "HIGH" priority papers not in Week 1
- Focus on Chapter 3 and Chapter 4 papers

### Week 3: Medium Priority & Gap Filling (20 papers)
- Remaining "MEDIUM" priority papers
- Any newly identified gaps

### Week 4: Final Review & Citation Check
- Verify all citations are correct
- Ensure BibTeX entries are complete
- Cross-reference with thesis chapters

---

## Verification Checklist

- [ ] All directories created
- [ ] Download script executed
- [ ] No download failures (check log)
- [ ] BibTeX file generated
- [ ] Status tracker initialized
- [ ] Critical papers read and annotated
- [ ] Citations integrated into thesis draft
- [ ] Peer-reviewed status verified for all papers
- [ ] DOIs added where available

---

## Notes

1. **arXiv papers**: Always check if a peer-reviewed version exists at a conference
2. **OpenReview**: ICLR 2025 papers available at openreview.net
3. **ACL Anthology**: Use aclanthology.org for ACL/EMNLP/NAACL papers
4. **Semantic Scholar**: Use for citation counts and related papers
5. **Connected Papers**: Visualize paper relationships

---

*Document generated: January 18, 2026*
*Total papers: 103 unique references*
*Estimated reading time: 80-100 hours*
