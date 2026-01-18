# Multi-Agent Software Engineering with Small Language Models
## Consolidated Master's Thesis Research & Implementation Framework

**Author:** Victor Sattamini  
**Date:** January 18, 2026  
**Version:** 1.0

---

## Executive Summary

This document consolidates all research materials, academic literature, and implementation plans for a Master's thesis on **multi-agent software engineering systems using Small Language Models (SLMs)** optimized for consumer-grade hardware (RTX 4070 8GB). It bridges the gap between:

1. **Academic foundations** — Peer-reviewed literature establishing theoretical scaffolding
2. **Technical implementation** — Practical system architecture for RTX 4070 constraints
3. **Evaluation methodology** — SWE-bench and related benchmarks for empirical validation

---

## Part I: Research Foundation Mapping

### 1.1 Document Inventory

| Document | Type | Content | Thesis Role |
|----------|------|---------|-------------|
| Implementation Guide (Oct 2025) | Technical | Model recommendations, frameworks, tools, arXiv papers | Implementation reference |
| Agent Swarms: SLM Research Review | Bibliographical | 38 citations, swarm architectures, orchestration patterns | Literature review draft |
| Supplementary Academic Literature | Academic | 80+ peer-reviewed papers organized by thesis section | Citation reinforcement |
| Implementation Plan | Engineering | 10-task TDD plan for local multi-agent system | Prototype development |

### 1.2 Thesis Structure Alignment

```
THESIS CHAPTERS                          SUPPORTING MATERIALS
─────────────────────────────────────────────────────────────────
Ch 1: Introduction                       
  └─ Paradigm shift argument             ← SLM Research Review §1.1-1.3
  └─ Research questions                  ← SLM Research Review §1.3
  
Ch 2: Foundational Principles            
  └─ Swarm intelligence                  ← SLM Research Review §2.1-2.3
  └─ SLMs vs LLMs                        ← Supplementary Lit (Phi, Gemma, Mistral papers)
  └─ Quantization theory                 ← Supplementary Lit (GPTQ, AWQ, QLoRA)
  
Ch 3: Architectural Patterns             
  └─ Orchestration mechanisms            ← SLM Research Review §3.1-3.4
  └─ Agent communication                 ← Implementation Guide Part 2
  └─ Reasoning paradigms                 ← Supplementary Lit (ReAct, CoT, ToT)
  
Ch 4: State of the Art                   
  └─ Framework analysis                  ← SLM Research Review §3.4
  └─ Software engineering agents         ← Implementation Guide Part 3
  └─ Recent advances                     ← Supplementary Lit (2024-2026 papers)
  
Ch 5: Evaluation & Benchmarking          
  └─ Benchmark landscape                 ← SLM Research Review §5.1-5.2
  └─ SWE-bench ecosystem                 ← Implementation Guide Part 3
  └─ Metrics design                      ← Supplementary Lit (AgentBench, WebArena)
  
Ch 6: Implementation                     
  └─ System architecture                 ← Implementation Plan Tasks 1-5
  └─ Memory management                   ← Implementation Plan Task 6
  └─ SWE-bench integration               ← Implementation Plan Tasks 7-8
  
Ch 7: Experimental Results               
  └─ Benchmark execution                 ← Implementation Plan Task 10
  └─ Comparative analysis                ← (To be generated)
  
Ch 8: Discussion & Future Work           
  └─ Challenges identified               ← SLM Research Review §6.1-6.2
  └─ Research directions                 ← SLM Research Review §7.2
```

---

## Part II: Academic Literature Synthesis

### 2.1 Publication Status of Existing Citations

Your SLM Research Review document contains 38 citations. After verification:

| Status | Count | Action Required |
|--------|-------|-----------------|
| Peer-reviewed | 4 | Keep as primary citations |
| Preprints with venues | 6 | Note venue, cite with arXiv ID |
| Industry blogs/docs | 22 | Replace with academic sources |
| Broken/inaccessible | 6 | Find alternatives |

**Critical gaps identified:**
- Section 2 (Foundations): Only 2 academic papers, needs 15-20
- Section 4 (SOTA): Heavy industry sources, needs academic grounding
- Section 5 (Evaluation): Benchmark papers cited but not formally

### 2.2 Priority Academic Papers by Thesis Section

#### Chapter 2: Foundational Principles

**SLM Technical Reports (Essential):**
```
1. Phi-3 Technical Report
   Authors: Abdin et al. (Microsoft)
   arXiv: 2404.14219, April 2024
   Status: Technical report
   Relevance: Core SLM capabilities, 3.8B matching Mixtral 8×7B
   
2. Phi-4 Technical Report  
   Authors: Abdin et al. (Microsoft)
   arXiv: 2412.08905, December 2024
   Status: Technical report
   Relevance: 14B model surpassing GPT-4 on STEM

3. Gemma 2 Technical Report
   Authors: Gemma Team (Google DeepMind)
   arXiv: 2408.00118, August 2024
   Status: Technical report
   Relevance: Open-weights SLM family, efficiency focus

4. Mistral 7B
   Authors: Jiang et al. (Mistral AI)
   arXiv: 2310.06825, October 2023
   Status: Preprint (~2,800 citations)
   Relevance: Sliding window attention, GQA innovations

5. Qwen2.5 Technical Report
   Authors: Qwen Team (Alibaba)
   arXiv: 2412.15115, December 2024
   Status: Technical report
   Relevance: SOTA open model, 18T token pretraining

6. TinyLLaMA
   Authors: Zhang et al.
   arXiv: 2401.02385, January 2024
   Status: Preprint
   Relevance: 1.1B model, over-training strategies for SLMs
```

**Quantization (Peer-Reviewed):**
```
7. GPTQ: Accurate Post-Training Quantization
   Authors: Frantar et al.
   Venue: ICLR 2023
   arXiv: 2210.17323
   Citations: ~1,500
   Relevance: One-shot 3-4 bit quantization

8. AWQ: Activation-aware Weight Quantization
   Authors: Lin et al.
   Venue: MLSys 2024 (Best Paper Award)
   arXiv: 2306.00978
   Relevance: Hardware-friendly, TinyChat

9. QLoRA: Efficient Finetuning of Quantized LLMs
   Authors: Dettmers et al.
   Venue: NeurIPS 2023
   arXiv: 2305.14314
   Citations: ~4,000
   Relevance: 4-bit NormalFloat, single-GPU finetuning

10. SmoothQuant
    Authors: Xiao et al.
    Venue: ICML 2023
    arXiv: 2211.10438
    Relevance: W8A8 quantization, 1.56× speedup
```

#### Chapter 3: Architectural Patterns

**Agent Reasoning (Peer-Reviewed Foundations):**
```
11. ReAct: Synergizing Reasoning and Acting
    Authors: Yao et al.
    Venue: ICLR 2023
    arXiv: 2210.03629
    Citations: ~3,500
    Relevance: FOUNDATIONAL - interleaved reasoning + action

12. Chain-of-Thought Prompting
    Authors: Wei et al.
    Venue: NeurIPS 2022
    arXiv: 2201.11903
    Citations: ~15,000
    Relevance: FOUNDATIONAL - intermediate reasoning steps

13. Toolformer
    Authors: Schick et al.
    Venue: NeurIPS 2023
    arXiv: 2302.04761
    Citations: ~2,500
    Relevance: Self-supervised tool learning

14. Tree-of-Thoughts
    Authors: Yao et al.
    Venue: NeurIPS 2023
    arXiv: 2305.10601
    Citations: ~2,200
    Relevance: BFS/DFS exploration, 74% on Game of 24

15. Self-Consistency
    Authors: Wang et al.
    Venue: ICLR 2023
    arXiv: 2203.11171
    Citations: ~3,500
    Relevance: Sample-and-marginalize decoding

16. Reflexion
    Authors: Shinn et al.
    Venue: NeurIPS 2023
    arXiv: 2303.11366
    Citations: ~1,500
    Relevance: Verbal reinforcement learning, 91% HumanEval
```

#### Chapter 4: State of the Art

**Multi-Agent Frameworks (Peer-Reviewed):**
```
17. MetaGPT
    Authors: Hong et al.
    Venue: ICLR 2024 (Oral, top 1.2%)
    arXiv: 2308.00352
    Citations: ~700
    Relevance: SOPs for structured workflows, 85.9% HumanEval

18. AutoGen
    Authors: Wu et al. (Microsoft Research)
    Venue: COLM 2024
    arXiv: 2308.08155
    Citations: ~1,500
    Relevance: Conversation programming paradigm

19. ChatDev
    Authors: Qian et al. (Tsinghua)
    Venue: ACL 2024
    arXiv: 2307.07924
    DOI: 10.18653/v1/2024.acl-long.810
    Citations: ~500
    Relevance: Chat chain, <7 min software generation

20. CAMEL
    Authors: Li et al. (KAUST)
    Venue: NeurIPS 2023
    arXiv: 2303.17760
    Citations: ~335
    Relevance: Role-playing paradigm, inception prompting

21. AgentVerse
    Authors: Chen et al. (Tsinghua)
    Venue: ICLR 2024
    arXiv: 2308.10848
    Citations: ~300
    Relevance: Dynamic agent composition
```

**Software Engineering Agents:**
```
22. SWE-agent
    Authors: Yang et al.
    Venue: NeurIPS 2024
    arXiv: 2405.15793
    Relevance: Agent-Computer Interface, 12.47% SWE-bench

23. Multi-Agent Debate
    Authors: Du et al. (MIT, DeepMind)
    Venue: ICML 2024
    arXiv: 2305.14325
    Citations: ~450
    Relevance: Consensus through debate
```

#### Chapter 5: Evaluation & Benchmarking

**Benchmark Papers (Peer-Reviewed):**
```
24. SWE-bench
    Authors: Jimenez et al.
    Venue: ICLR 2024 (Oral)
    arXiv: 2310.06770
    Relevance: PRIMARY BENCHMARK - 2,294 GitHub issues

25. AgentBench
    Authors: Liu et al.
    Venue: ICLR 2024
    arXiv: 2308.03688
    Citations: ~500
    Relevance: 8 environments, 29 LLMs evaluated

26. WebArena
    Authors: Zhou et al.
    Venue: ICLR 2024
    arXiv: 2307.13854
    Citations: ~400
    Relevance: 812 web tasks, GPT-4 at 14.41%

27. GAIA
    Authors: Mialon et al. (Meta-FAIR, HuggingFace)
    arXiv: 2311.12983
    Citations: ~200
    Relevance: 466 real-world questions, GPT-4+plugins at 15%

28. MINT
    Authors: Wang et al. (UIUC)
    Venue: ICLR 2024
    arXiv: 2309.10691
    Relevance: Multi-turn tool use evaluation

29. ToolLLM/ToolBench
    Authors: Qin et al. (OpenBMB)
    Venue: ICLR 2024 (Spotlight)
    arXiv: 2307.16789
    Citations: ~300
    Relevance: 16,464 REST APIs

30. EvalPlus
    Authors: Liu et al.
    Venue: NeurIPS 2023
    arXiv: 2305.01210
    Relevance: 80× more test cases for HumanEval
```

#### Cross-Cutting: RAG & Retrieval

```
31. RAG (Original)
    Authors: Lewis et al.
    Venue: NeurIPS 2020
    arXiv: 2005.11401
    DOI: 10.5555/3495724.3496517
    Relevance: FOUNDATIONAL - parametric + non-parametric memory

32. Dense Passage Retrieval (DPR)
    Authors: Karpukhin et al.
    Venue: EMNLP 2020
    arXiv: 2004.04906
    DOI: 10.18653/v1/2020.emnlp-main.550
    Relevance: Dual-encoder dense retrieval

33. ColBERT
    Authors: Khattab & Zaharia
    Venue: SIGIR 2020
    arXiv: 2004.12832
    DOI: 10.1145/3397271.3401075
    Relevance: Late interaction, MaxSim

34. Self-RAG
    Authors: Asai et al.
    Venue: ICLR 2024
    arXiv: 2310.11511
    Relevance: Adaptive retrieval with reflection tokens
```

### 2.3 Citation Quality Assessment

| Thesis Section | Current Citations | Academic Need | Gap |
|----------------|-------------------|---------------|-----|
| Ch 2: Foundations | 8 | 20-25 | +12-17 |
| Ch 3: Architectures | 12 | 15-20 | +3-8 |
| Ch 4: SOTA | 15 | 20-25 | +5-10 |
| Ch 5: Evaluation | 10 | 15-20 | +5-10 |
| Total | 45 | 70-90 | +25-45 |

---

## Part III: Implementation Plan Analysis & Revisions

### 3.1 Alignment with Academic Framework

The implementation plan provides a solid TDD-based approach. However, several modifications are recommended to better align with thesis requirements:

#### Task 1-2: Project Structure & Agent Communication

**Current:** Basic Python project with Pydantic models
**Recommended additions:**
- Add explicit support for MCP (Model Context Protocol) — emerging standard
- Include structured logging for experiment reproducibility
- Add configuration for different orchestration patterns (centralized vs distributed)

```python
# Suggested addition to src/agents/types.py
class OrchestrationPattern(str, Enum):
    CENTRALIZED = "centralized"       # Supervisor architecture
    HIERARCHICAL = "hierarchical"     # Planner + Executor pattern
    PEER_TO_PEER = "peer_to_peer"     # Handoff orchestration
    HYBRID = "hybrid"                 # Sequential with feedback loops
```

#### Task 3: Model Interface

**Critical revision needed:** The current placeholder doesn't account for:
1. **Quantization formats** — GGUF, GPTQ, AWQ support
2. **Memory-efficient loading** — Flash attention, 4-bit inference
3. **Model switching overhead** — Sequential execution timing

```python
# Suggested ModelInterface enhancement
class ModelConfig(BaseModel):
    name: str
    path: str
    quantization: Literal["fp16", "int8", "int4", "gptq", "awq", "gguf"]
    context_length: int = 4096
    vram_requirement_gb: float
    
class LocalSLMModel(ModelInterface):
    async def load(self, config: ModelConfig) -> None:
        # Check VRAM before loading
        available_vram = self._get_available_vram()
        if config.vram_requirement_gb > available_vram:
            raise MemoryError(f"Need {config.vram_requirement_gb}GB, have {available_vram}GB")
        
        # Load with appropriate backend
        if config.quantization == "gguf":
            self._load_with_llama_cpp(config)
        elif config.quantization in ["gptq", "awq"]:
            self._load_with_exllamav2(config)
        else:
            self._load_with_transformers(config)
```

#### Task 4: Coordinator Agent

**Enhancement for thesis:** Add support for different decomposition strategies based on literature:

```python
class DecompositionStrategy(str, Enum):
    SEQUENTIAL = "sequential"      # Linear pipeline (Task 1 → 2 → 3)
    PARALLEL = "parallel"          # Independent subtasks
    ITERATIVE = "iterative"        # Feedback loops (like Reflexion)
    HIERARCHICAL = "hierarchical"  # Tree-structured decomposition

class TaskDecomposer:
    def __init__(self, strategy: DecompositionStrategy = DecompositionStrategy.SEQUENTIAL):
        self.strategy = strategy
        
    async def decompose(self, task: Task) -> List[Task]:
        if self.strategy == DecompositionStrategy.ITERATIVE:
            return await self._decompose_with_refinement(task)
        elif self.strategy == DecompositionStrategy.HIERARCHICAL:
            return await self._decompose_hierarchical(task)
        # ... etc
```

#### Task 5: Specialist Agents

**Academic grounding needed:** Map agents to literature:

| Agent | Academic Basis | Key Paper |
|-------|---------------|-----------|
| CodeGenerator | Program synthesis | Codex (Chen et al., 2021) |
| TestWriter | Test generation | EvoSuite principles |
| Debugger | Automated program repair | APR survey (Xia et al., 2023) |
| Reviewer | Code review automation | CodeReviewer (Li et al., 2022) |

**Suggested addition:** Self-debugging capability based on Reflexion:

```python
class SelfDebuggingCodeGenerator(CodeGenerator):
    """Code generator with self-reflection loop (Shinn et al., 2023)"""
    
    MAX_REFINEMENT_ITERATIONS = 3
    
    async def process_task(self, task: Task) -> str:
        code = await self._generate_initial_code(task)
        
        for iteration in range(self.MAX_REFINEMENT_ITERATIONS):
            # Test the code
            test_result = await self._run_tests(code)
            
            if test_result.all_passed:
                return code
                
            # Reflect on failure
            reflection = await self._reflect_on_failure(code, test_result)
            
            # Generate improved code
            code = await self._refine_code(code, reflection)
            
        return code  # Return best effort
```

#### Task 6: Memory Management

**Enhancement:** Add GPU memory profiling for empirical analysis:

```python
class GPUMemoryProfiler:
    """Track GPU memory for thesis experiments"""
    
    def __init__(self):
        self.measurements: List[MemoryMeasurement] = []
        
    def measure(self, label: str) -> MemoryMeasurement:
        import torch
        measurement = MemoryMeasurement(
            label=label,
            timestamp=time.time(),
            allocated_gb=torch.cuda.memory_allocated() / 1e9,
            reserved_gb=torch.cuda.memory_reserved() / 1e9,
            max_allocated_gb=torch.cuda.max_memory_allocated() / 1e9
        )
        self.measurements.append(measurement)
        return measurement
        
    def export_for_thesis(self, path: str) -> None:
        """Export measurements for thesis figures"""
        df = pd.DataFrame([m.dict() for m in self.measurements])
        df.to_csv(path, index=False)
```

#### Task 7: SWE-bench Integration

**Critical enhancement:** Use actual SWE-bench Lite dataset:

```python
class SWEBenchEvaluator:
    SWEBENCH_LITE_URL = "https://raw.githubusercontent.com/princeton-nlp/SWE-bench/main/swebench/test.json"
    
    async def load_tasks(self, lite: bool = True) -> List[SWEBenchTask]:
        if lite:
            # Load actual SWE-bench Lite (300 instances)
            response = await self._fetch_dataset(self.SWEBENCH_LITE_URL)
            return [self._parse_task(item) for item in response[:300]]
        else:
            # Full dataset (2,294 instances)
            return await self._load_full_swebench()
            
    async def evaluate_with_harness(self, task: SWEBenchTask, patch: str) -> Dict:
        """Use official SWE-bench evaluation harness"""
        # Apply patch to repository
        repo = await self._setup_repository(task)
        await self._apply_patch(repo, patch)
        
        # Run test suite
        results = await self._run_tests(repo, task.test_patch)
        
        return {
            "task_id": task.id,
            "resolved": results.all_passed,
            "passed_tests": results.passed,
            "failed_tests": results.failed,
            "error_log": results.error_log
        }
```

### 3.2 Additional Tasks Recommended

**Task 11: Experiment Runner for Thesis**
```python
class ThesisExperimentRunner:
    """Run reproducible experiments for thesis chapters"""
    
    async def run_ablation_study(self) -> Dict:
        """Compare different configurations"""
        configs = [
            {"orchestration": "centralized", "model": "qwen2.5-coder-7b"},
            {"orchestration": "hierarchical", "model": "qwen2.5-coder-7b"},
            {"orchestration": "centralized", "model": "deepseek-coder-7b"},
            # ... more configurations
        ]
        
        results = []
        for config in configs:
            system = MultiAgentSystem(**config)
            metrics = await system.run_evaluation(lite=True)
            results.append({**config, **metrics})
            
        return results
        
    async def run_memory_study(self) -> Dict:
        """Measure memory usage across configurations"""
        profiler = GPUMemoryProfiler()
        # ... measurement logic
```

**Task 12: Baseline Comparisons**
```python
class BaselineComparator:
    """Compare multi-agent system against baselines"""
    
    async def compare_single_vs_multi_agent(self, tasks: List[Task]) -> Dict:
        """Key comparison for thesis: single agent vs swarm"""
        
        # Single agent baseline
        single_agent = CodeGenerator("single", "qwen2.5-coder-7b")
        single_results = [await single_agent.process_task(t) for t in tasks]
        
        # Multi-agent system
        system = MultiAgentSystem()
        multi_results = [await system.process_task(t.description) for t in tasks]
        
        return {
            "single_agent": self._compute_metrics(single_results),
            "multi_agent": self._compute_metrics(multi_results),
            "improvement": self._compute_improvement(single_results, multi_results)
        }
```

### 3.3 Revised Implementation Timeline

| Week | Tasks | Thesis Chapter Support |
|------|-------|------------------------|
| 1 | Tasks 1-3 (Infrastructure) | Ch 6 foundation |
| 2 | Tasks 4-5 (Agents) | Ch 3 examples |
| 3 | Task 6 (Memory) + Profiling | Ch 6 analysis |
| 4 | Task 7-8 (SWE-bench) | Ch 5 methodology |
| 5 | Task 9-10 (Integration) | Ch 6 completion |
| 6 | Task 11-12 (Experiments) | Ch 7 results |
| 7 | Analysis & Writing | Ch 7-8 |

---

## Part IV: Model Recommendations for RTX 4070 8GB

### 4.1 Validated Model Stack

Based on VRAM constraints and benchmark performance:

| Role | Primary Model | VRAM (Q5_K_M) | HumanEval | Alternative |
|------|--------------|---------------|-----------|-------------|
| Code Generation | Qwen2.5-Coder-7B-Instruct | 5.7 GB | 84.1% | DeepSeek-Coder-7B |
| Reasoning | Llama 3.1 8B Instruct | 5.9 GB | 72.6% | Phi-3 Mini 128K |
| Lightweight | Qwen2.5-Coder-3B-Instruct | 2.4 GB | 65.8% | StarCoder2-3B |

### 4.2 Sequential Loading Strategy

```python
# Optimal loading order for 8GB VRAM
LOADING_SEQUENCE = [
    ("coordinator", "qwen2.5-coder-3b", 2.4),   # Load first, keep resident
    ("code_gen", "qwen2.5-coder-7b", 5.7),      # Load for code tasks
    ("test_writer", "qwen2.5-coder-7b", 0),     # Reuse code_gen model
    ("reviewer", "llama-3.1-8b", 5.9),          # Swap for review
]

async def execute_with_memory_management(system: MultiAgentSystem, task: Task):
    # Coordinator always loaded (small model)
    subtasks = await system.coordinator.decompose_task(task)
    
    results = []
    for subtask in subtasks:
        # Load appropriate model for subtask
        required_model = AGENT_MODEL_MAP[subtask.type]
        if required_model != system.model_manager.current_model:
            await system.model_manager.swap_model(required_model)
            
        result = await system.execute_subtask(subtask)
        results.append(result)
        
    return results
```

---

## Part V: Evaluation Strategy

### 5.1 Primary Benchmark: SWE-bench Lite

**Dataset characteristics:**
- 300 instances (subset of full 2,294)
- 12 Python repositories
- Real GitHub issues with patches
- Difficulty: Moderate to challenging

**Evaluation metrics:**
```python
@dataclass
class SWEBenchMetrics:
    resolved_rate: float          # % of issues resolved
    avg_patch_similarity: float   # BLEU/CodeBLEU with gold patches
    avg_test_pass_rate: float     # % tests passing
    avg_tokens_generated: int     # Efficiency metric
    avg_time_per_task: float      # Latency metric
    memory_peak_gb: float         # Resource usage
```

### 5.2 Secondary Benchmarks

| Benchmark | Purpose | Thesis Section |
|-----------|---------|----------------|
| HumanEval | Code generation baseline | Ch 5 |
| MBPP | Entry-level Python tasks | Ch 5 |
| AgentBench (subset) | Multi-env evaluation | Ch 5 |

### 5.3 Ablation Studies for Thesis

1. **Single vs Multi-Agent:** Core thesis hypothesis
2. **Orchestration patterns:** Centralized vs hierarchical vs peer-to-peer
3. **Model size trade-offs:** 3B vs 7B vs mixed
4. **Quantization impact:** FP16 vs INT8 vs INT4

---

## Part VI: Research Questions Mapping

### RQ1: Technical and Economic Arguments for SLM Swarms

**Literature support:**
- Phi-3 showing 3.8B matching Mixtral 8×7B (data quality > scale)
- TinyLLaMA demonstrating over-training strategies
- AWQ/GPTQ enabling efficient quantization

**Implementation evidence:**
- Memory profiler data showing VRAM efficiency
- Cost analysis: local inference vs API costs

### RQ2: Architectural Patterns for Specialized Agents

**Literature support:**
- MetaGPT SOPs for structured workflows
- AutoGen conversation programming
- ReAct for reasoning-action interleaving

**Implementation evidence:**
- Coordinator + specialist agent design
- Task decomposition strategies
- Communication protocol analysis

### RQ3: Benchmarks and Evaluation Methodologies

**Literature support:**
- SWE-bench ecosystem (original, Pro, Live)
- AgentBench multi-environment evaluation
- EvalPlus test augmentation

**Implementation evidence:**
- SWE-bench Lite results
- Custom metrics for swarm evaluation
- Memory/latency trade-off analysis

### RQ4: Challenges and Future Directions

**Literature support:**
- Emergent behavior studies (SLM Research Review §2.3)
- Scalability challenges (§6.1-6.2)
- Interpretability concerns (§6.2)

**Implementation evidence:**
- Observed failure modes
- Memory bottlenecks
- Coordination overhead analysis

---

## Part VII: Gap Analysis and Next Steps

### 7.1 Literature Gaps to Address

| Gap | Current State | Action | Priority |
|-----|---------------|--------|----------|
| MCP academic papers | 1 preprint | Search for formal spec work | Medium |
| Edge deployment | Limited | Survey mobile/embedded LLM work | Low |
| Multi-agent emergent behavior | Descriptive | Find formal analysis papers | High |
| SLM-specific benchmarks | None identified | May need to acknowledge gap | High |

### 7.2 Implementation Gaps

| Gap | Current State | Action | Priority |
|-----|---------------|--------|----------|
| Real model integration | Placeholder | Implement GGUF/ExLlamaV2 loading | Critical |
| Actual SWE-bench harness | Mock data | Integrate official evaluation | Critical |
| Memory profiling | Not implemented | Add GPU monitoring | High |
| Experiment reproducibility | Basic | Add seeding, logging, configs | Medium |

### 7.3 Recommended Immediate Actions

1. **Week 1:** Finalize literature citations, update bibliography
2. **Week 2:** Begin implementation with real model loading
3. **Week 3:** Integrate SWE-bench Lite evaluation harness
4. **Week 4:** Run initial experiments, validate methodology
5. **Week 5:** Conduct ablation studies
6. **Week 6:** Write results and analysis

---

## Appendix A: Complete Citation List by Section

*See separate document: "Academic Literature Download & Processing Plan"*

## Appendix B: Implementation Code Templates

*See implementation plan with suggested modifications above*

## Appendix C: Configuration Files

```yaml
# config/thesis_experiment.yaml
experiment:
  name: "multi-agent-swe-bench-evaluation"
  seed: 42
  
models:
  coordinator:
    name: "qwen2.5-coder-3b-instruct"
    quantization: "q5_k_m"
    vram_gb: 2.4
    
  code_generator:
    name: "qwen2.5-coder-7b-instruct"
    quantization: "q5_k_m"
    vram_gb: 5.7
    
  reviewer:
    name: "llama-3.1-8b-instruct"
    quantization: "q5_k_m"
    vram_gb: 5.9

orchestration:
  pattern: "hierarchical"
  max_iterations: 3
  feedback_enabled: true

evaluation:
  benchmark: "swebench-lite"
  subset_size: 300
  metrics:
    - resolved_rate
    - test_pass_rate
    - tokens_generated
    - time_per_task
    - memory_peak_gb

hardware:
  gpu: "RTX 4070"
  vram_gb: 8
  system_ram_gb: 32
```

---

*Document generated: January 18, 2026*
*For: Victor Sattamini - Master's Thesis*
