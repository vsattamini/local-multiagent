# System Architecture: Context-Accumulation Swarm

## Design Philosophy

**Core Principle**: Agents are not separate model instances—they are **different context windows** attached to a single shared model. This is:
- Memory efficient (one model loaded)
- Conceptually clean (tests pure context-based emergence)
- Fast (no model swapping overhead)

```
┌─────────────────────────────────────────────────────────────────┐
│                         SHARED MODEL                            │
│                    (Qwen2.5-Coder, frozen)                      │
└─────────────────────────────────────────────────────────────────┘
                              │
           ┌──────────────────┼──────────────────┐
           │                  │                  │
           ▼                  ▼                  ▼
    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
    │  Agent 0    │    │  Agent 1    │    │  Agent 2    │
    │  Context    │    │  Context    │    │  Context    │
    │  + History  │    │  + History  │    │  + History  │
    └─────────────┘    └─────────────┘    └─────────────┘
```

---

## High-Level Components

```
┌────────────────────────────────────────────────────────────────────┐
│                           SWARM SYSTEM                             │
├────────────────────────────────────────────────────────────────────┤
│                                                                    │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐       │
│  │   ROUTER     │────▶│   AGENTS     │────▶│  EXECUTOR    │       │
│  │              │     │   (Contexts) │     │              │       │
│  │ - Affinity   │     │              │     │ - Run tests  │       │
│  │ - Selection  │     │ - Few-shot   │     │ - Capture    │       │
│  │              │     │ - History    │     │   output     │       │
│  └──────────────┘     └──────────────┘     └──────────────┘       │
│         │                    │                    │                │
│         │                    │                    │                │
│         ▼                    ▼                    ▼                │
│  ┌─────────────────────────────────────────────────────────┐      │
│  │                      FEEDBACK LOOP                       │      │
│  │                                                          │      │
│  │   success ──▶ add to context, increase affinity          │      │
│  │   failure ──▶ (optionally log), decrease affinity        │      │
│  └─────────────────────────────────────────────────────────┘      │
│                              │                                     │
│                              ▼                                     │
│  ┌─────────────────────────────────────────────────────────┐      │
│  │                     METRICS ENGINE                       │      │
│  │                                                          │      │
│  │   - Specialization Index (S)                             │      │
│  │   - Context Divergence (D)                               │      │
│  │   - Functional Differentiation (F)                       │      │
│  │   - Timeline snapshots                                   │      │
│  └─────────────────────────────────────────────────────────┘      │
│                              │                                     │
│                              ▼                                     │
│  ┌─────────────────────────────────────────────────────────┐      │
│  │                       LOGGER                             │      │
│  │                                                          │      │
│  │   - Task-level logs (JSON)                               │      │
│  │   - Periodic snapshots                                   │      │
│  │   - Final results                                        │      │
│  └─────────────────────────────────────────────────────────┘      │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
```

---

## Module Breakdown

### 1. Model Interface (`model.py`)

```python
class ModelInterface:
    """Wrapper around the LLM. Single instance, shared by all agents."""
    
    def __init__(self, model_path: str, context_length: int = 4096):
        self.model = load_model(model_path)  # llama-cpp-python or vLLM
        self.context_length = context_length
    
    def generate(self, prompt: str, max_tokens: int = 512) -> str:
        """Generate completion for prompt."""
        pass
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        pass
```

**Key Decision**: Use `llama-cpp-python` for simplicity and low VRAM usage, or `vLLM` if you need faster batch inference later.

---

### 2. Agent State (`agent.py`)

```python
@dataclass
class Agent:
    """An agent is just a context window + metadata. No separate model."""
    
    id: int
    context_buffer: list[tuple[str, str]]  # (problem, solution) pairs
    max_context_examples: int = 5
    task_history: dict[str, list[bool]] = field(default_factory=dict)
    
    def add_success(self, problem: str, solution: str, task_type: str):
        """Add successful example to context."""
        self.context_buffer.append((problem, solution))
        if len(self.context_buffer) > self.max_context_examples:
            self.context_buffer.pop(0)  # FIFO
        
        # Track history
        if task_type not in self.task_history:
            self.task_history[task_type] = []
        self.task_history[task_type].append(True)
    
    def add_failure(self, task_type: str):
        """Record failure (no context addition)."""
        if task_type not in self.task_history:
            self.task_history[task_type] = []
        self.task_history[task_type].append(False)
    
    def build_prompt(self, system_prompt: str, new_problem: str) -> str:
        """Construct full prompt with few-shot examples."""
        prompt = system_prompt + "\n\n"
        for prob, sol in self.context_buffer:
            prompt += f"Problem:\n{prob}\n\nSolution:\n```python\n{sol}\n```\n\n"
        prompt += f"Problem:\n{new_problem}\n\nSolution:\n```python\n"
        return prompt
    
    def success_rate(self, task_type: str) -> float:
        """Get success rate for task type."""
        history = self.task_history.get(task_type, [])
        if not history:
            return 0.5  # Prior
        return sum(history) / len(history)
```

---

### 3. Router (`router.py`)

```python
class AffinityRouter:
    """Routes tasks to agents based on learned affinities."""
    
    def __init__(self, agents: list[Agent], task_types: list[str]):
        self.agents = agents
        self.task_types = task_types
        self.temperature = 0.5
    
    def select_agent(self, task_type: str) -> Agent:
        """Select agent using softmax over success rates."""
        scores = [agent.success_rate(task_type) for agent in self.agents]
        probs = self._softmax(scores)
        idx = np.random.choice(len(self.agents), p=probs)
        return self.agents[idx]
    
    def _softmax(self, scores: list[float]) -> np.ndarray:
        scores = np.array(scores) / self.temperature
        exp_scores = np.exp(scores - np.max(scores))
        return exp_scores / exp_scores.sum()
```

**Variants to Implement Later**:
- `RandomRouter` (baseline)
- `RoundRobinRouter` (control)
- `GreedyRouter` (always pick best)
- `UCBRouter` (exploration bonus for less-tried agents)

---

### 4. Executor (`executor.py`)

```python
class CodeExecutor:
    """Execute generated code against test cases."""
    
    def __init__(self, timeout: int = 5):
        self.timeout = timeout
    
    def execute(self, code: str, test_code: str) -> ExecutionResult:
        """Run code against tests in sandboxed environment."""
        # Use subprocess or Docker for isolation
        pass

@dataclass
class ExecutionResult:
    success: bool
    stdout: str
    stderr: str
    execution_time: float
```

**Safety Note**: Always sandbox code execution. Use Docker or at minimum `subprocess` with timeout.

---

### 5. Metrics Engine (`metrics.py`)

```python
class MetricsEngine:
    """Compute all emergence metrics."""
    
    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        self.embedder = SentenceTransformer(embedding_model)
    
    def specialization_index(self, task_log: list[dict]) -> float:
        """Compute S from task log."""
        # Implementation from metrics document
        pass
    
    def context_divergence(self, agents: list[Agent]) -> float:
        """Compute D from agent contexts."""
        # Implementation from metrics document
        pass
    
    def functional_differentiation(self, task_log: list[dict]) -> dict:
        """Compute chi-square test results."""
        # Implementation from metrics document
        pass
    
    def snapshot(self, agents: list[Agent], task_log: list[dict]) -> dict:
        """Capture full system state for logging."""
        return {
            "S": self.specialization_index(task_log),
            "D": self.context_divergence(agents),
            "contexts": {a.id: a.context_buffer for a in agents},
            "task_count": len(task_log)
        }
```

---

### 6. Experiment Runner (`experiment.py`)

```python
class Experiment:
    """Main experiment orchestrator."""
    
    def __init__(self, config: ExperimentConfig):
        self.model = ModelInterface(config.model_path)
        self.agents = [Agent(id=i) for i in range(config.n_agents)]
        self.router = AffinityRouter(self.agents, config.task_types)
        self.executor = CodeExecutor()
        self.metrics = MetricsEngine()
        self.logger = ExperimentLogger(config.output_dir)
        self.config = config
    
    def run(self, tasks: list[Task]):
        """Run full experiment."""
        for i, task in enumerate(tasks):
            # 1. Route
            agent = self.router.select_agent(task.task_type)
            
            # 2. Generate
            prompt = agent.build_prompt(self.config.system_prompt, task.problem)
            solution = self.model.generate(prompt)
            
            # 3. Execute
            result = self.executor.execute(solution, task.test_code)
            
            # 4. Update
            if result.success:
                agent.add_success(task.problem, solution, task.task_type)
            else:
                agent.add_failure(task.task_type)
            
            # 5. Log
            self.logger.log_task(task, agent, result)
            
            # 6. Periodic snapshot
            if (i + 1) % self.config.snapshot_interval == 0:
                snapshot = self.metrics.snapshot(self.agents, self.logger.task_log)
                self.logger.log_snapshot(snapshot)
        
        # Final analysis
        self.logger.finalize()
```

---

### 7. Configuration (`config.py`)

```python
@dataclass
class ExperimentConfig:
    # Model
    model_path: str = "models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf"
    context_length: int = 4096
    max_tokens: int = 512
    
    # Agents
    n_agents: int = 3
    max_context_examples: int = 5
    
    # Routing
    router_type: str = "affinity"  # or "random", "round_robin"
    router_temperature: float = 0.5
    
    # Tasks
    task_types: list[str] = field(default_factory=lambda: ["string", "math", "list", "logic"])
    
    # Experiment
    snapshot_interval: int = 10
    output_dir: str = "results/pilot_001"
    
    # Prompts
    system_prompt: str = "You are a Python coding assistant. Write clean, correct code."
```

---

## Directory Structure

```
local-multiagent/
├── src/
│   ├── __init__.py
│   ├── model.py          # LLM interface
│   ├── agent.py          # Agent state
│   ├── router.py         # Task routing
│   ├── executor.py       # Code execution
│   ├── metrics.py        # Emergence metrics
│   ├── experiment.py     # Main orchestrator
│   ├── config.py         # Configuration
│   └── logger.py         # Logging utilities
├── data/
│   ├── humaneval/        # Benchmark tasks
│   └── task_types.json   # Task categorization
├── models/
│   └── *.gguf            # Downloaded model files
├── results/
│   └── pilot_001/        # Experiment outputs
│       ├── task_log.jsonl
│       ├── snapshots.jsonl
│       └── final_metrics.json
├── notebooks/
│   └── analysis.ipynb    # Post-experiment analysis
├── tests/
│   └── test_*.py         # Unit tests
├── scripts/
│   ├── download_model.sh
│   ├── run_pilot.py
│   └── analyze_results.py
├── config/
│   └── pilot.yaml        # Experiment configs
├── requirements.txt
└── README.md
```

---

## Data Flow

```
┌─────────────┐
│   TASK      │
│ (problem +  │
│  test_code) │
└──────┬──────┘
       │
       ▼
┌──────────────┐      ┌──────────────┐
│   ROUTER     │─────▶│   AGENT      │
│ select_agent │      │ (context)    │
└──────────────┘      └──────┬───────┘
                             │
                             ▼
                      ┌──────────────┐
                      │    MODEL     │
                      │  generate()  │
                      └──────┬───────┘
                             │
                             ▼
                      ┌──────────────┐
                      │  EXECUTOR    │
                      │  execute()   │
                      └──────┬───────┘
                             │
              ┌──────────────┴──────────────┐
              │                             │
              ▼                             ▼
       ┌─────────────┐              ┌─────────────┐
       │  SUCCESS    │              │  FAILURE    │
       │             │              │             │
       │ • Add to    │              │ • Log only  │
       │   context   │              │ • Decrease  │
       │ • Increase  │              │   affinity  │
       │   affinity  │              │             │
       └──────┬──────┘              └──────┬──────┘
              │                             │
              └──────────────┬──────────────┘
                             │
                             ▼
                      ┌──────────────┐
                      │   LOGGER     │
                      │ log_task()   │
                      └──────────────┘
```

---

## Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Single model instance | Yes | Memory efficient, tests context-only emergence |
| Context as FIFO buffer | Yes | Simple, prevents unbounded growth |
| Affinity-based routing | Softmax | Allows exploration while favoring good matches |
| Task type labels | Manual | Avoids confounding with automatic clustering |
| Synchronous execution | Yes | Simpler for pilot; can parallelize later |
| JSONL logging | Yes | Append-only, easy to analyze |

---

## Extension Points

For later experiments, you can extend:

1. **Router variants**: UCB, Thompson sampling, learned routing
2. **Context strategies**: Category-weighted, recency-weighted, quality-weighted
3. **Negative examples**: Add failed attempts to context with "don't do this" framing
4. **Peer review**: Have agents critique each other's solutions before execution
5. **Hierarchical**: Add coordinator agent that delegates sub-tasks
6. **Parallel**: Run multiple agents simultaneously on batch

---

*Document Version: 1.0*
