# AI Research Paper Summaries — Q1 2026

Recommendation key: **⭐ READ IN FULL** = high relevance to ModelSight pipeline, harness engineering, LLM eval, or cost optimization. **📌 SKIM** = useful patterns or results worth knowing. **⚪ SKIP** = interesting but low direct relevance.

---

## THEME 1: HARNESS & WORKFLOW ENGINEERING

---

### Meta-Harness — ⭐ READ IN FULL
**Stanford / Wisconsin, March 2026** — [arXiv:2603.28052](https://arxiv.org/abs/2603.28052)

**Summary.** Introduces an outer-loop system that uses an agentic proposer (Claude Code) to automatically search over harness code for LLM applications. The proposer has full access to source code, scores, and execution traces of all prior candidates through a filesystem. Rather than tweaking prompts, it proposes structurally different harness designs and evaluates them end-to-end.

**Results.** +7.7 points over SOTA context management on online text classification with 4× fewer tokens. +4.7 points on 200 IMO-level math problems. Ranked #2 on TerminalBench-2 for Opus 4.6. Discovered harnesses transfer across 5 held-out models with consistent gains. The paper quantifies a 6× performance gap from harness changes alone on the same benchmark with the same model.

**Implications.** This is the strongest evidence yet that harness optimization is higher-leverage than model scaling for many applications. Directly relevant to ModelSight's multi-stage pipeline—the extraction, matching, and reconciliation harness around Claude calls may be leaving significant performance on the table. The transferability finding means a harness optimized on Sonnet could generalize to other models.

---

### Natural-Language Agent Harnesses (NLAHs) — ⭐ READ IN FULL
**March 2026** — [arXiv:2603.25723](https://arxiv.org/abs/2603.25723)

**Summary.** Proposes expressing harness behavior in editable natural language rather than controller code, with an Intelligent Harness Runtime (IHR) that executes these specifications through explicit contracts, durable artifacts, and lightweight adapters. Provides a code-to-text migration path for existing harnesses.

**Results.** Migrating from Python harnesses to NLAHs improved OSWorld performance from 30.4 to 47.2 on held-out tasks—a 55% gain from changing the harness representation format alone.

**Implications.** The portability and inspectability of NL-expressed harness logic is compelling for production systems. If ModelSight's pipeline logic were expressed as natural-language contracts rather than hardcoded Python, it becomes version-controllable, auditable, and modifiable without code changes. The IHR pattern of explicit contracts between stages maps cleanly onto extraction → matching → reconciliation flows.

---

### Anthropic Multi-Agent Harness Design Blog — ⭐ READ IN FULL
**Anthropic, March 2026** — [anthropic.com/engineering/harness-design-long-running-apps](https://www.anthropic.com/engineering/harness-design-long-running-apps)

**Summary.** Engineering blog describing a GAN-inspired three-agent architecture (Planner, Generator, Evaluator) for long-running autonomous software engineering. Separation of generation and evaluation into distinct agents with fresh context windows. Also covers a two-agent initializer/coder architecture modeled after engineering shift handoffs.

**Results.** A retro game built with the full harness demonstrated substantially better quality than solo attempts. Fresh context windows per iteration eliminated "context anxiety" where models prematurely wrap up tasks. Separating doing from judging was the strongest single lever for quality improvement. Cost: ~20× higher than single-agent.

**Implications.** The Planner/Generator/Evaluator separation maps directly onto ModelSight's pipeline: a planning stage that expands extraction specs, a generator that does KPI matching, and an evaluator that validates against reference data. The "context anxiety" finding explains degradation patterns seen in long extraction runs. Fresh context windows per stage should be tested.

---

### Workflow Optimization for LLM Agents (IBM Survey) — 📌 SKIM
**IBM, March 2026** — [arXiv:2603.22386](https://arxiv.org/abs/2603.22386)

**Summary.** Comprehensive survey mapping methods for designing and optimizing LLM agent workflows, formalized as Agentic Computation Graphs (ACGs). Organizes prior work along three dimensions: when structure is determined (design-time vs runtime), what is optimized (prompts, topology, tools), and which evaluation signals guide optimization.

**Results.** Covers AFlow (MCTS over operator graphs), Automated Design of Agentic Systems (code-space search via meta-agents), evolutionary multi-agent design, and more. Distinguishes reusable templates from run-specific realized graphs from execution traces.

**Implications.** Useful reference taxonomy. The ACG formalism provides language for reasoning about ModelSight's pipeline stages as a graph where optimization targets are node prompts, edge routing, and retrieval strategies. Worth skimming the categorization tables.

---

### Codified Context — ⭐ READ IN FULL
**February 2026** — [arXiv:2602.20478](https://arxiv.org/abs/2602.20478)

**Summary.** Presents a three-component context infrastructure developed during construction of a 108,000-line C# distributed system over 283 development sessions: (1) a hot-memory constitution encoding conventions and retrieval hooks, (2) 19 domain-expert agents each owning a bounded domain, (3) 34 cold-memory specification documents retrieved on demand.

**Results.** The knowledge-to-code ratio stabilized at ~24.2%. Session continuity improved across 283 sessions, preventing convention amnesia and repeated mistakes. The tiered approach keeps active context lean while ensuring detailed specs are accessible.

**Implications.** The three-tier pattern (always-loaded conventions → domain-specialist routing → on-demand deep specs) is directly applicable to structuring ModelSight's codebase for Claude Code-assisted development. The domain-expert agent concept maps to having specialized contexts for extraction vs. matching vs. reconciliation subsystems.

---

### Evaluating AGENTS.md — 📌 SKIM
**ETH Zurich / LogicStar, February 2026** — [arXiv:2602.11988](https://arxiv.org/abs/2602.11988)

**Summary.** Tests whether AGENTS.md files actually improve coding agent performance. Evaluates four agents (Claude Code, Codex, Qwen Code) across multiple configurations.

**Results.** Human-written AGENTS.md provided a modest +4% improvement; LLM-generated ones reduced success rates by -2%. Both increased inference cost by 20%+. Context files cause agents to explore more code paths, introducing noise that dilutes task-relevant information.

**Implications.** Counterintuitive but important: more context ≠ better performance for coding agents. Keep AGENTS.md and equivalent project context files minimal and focused on hard constraints. This aligns with the Codified Context paper's recommendation for tiered retrieval rather than dump-everything-in-context.

---

## THEME 2: MULTI-AGENT COORDINATION

---

### CAID: Asynchronous Software Engineering Agents — ⭐ READ IN FULL
**CMU, March 2026** — [arXiv:2603.21489](https://arxiv.org/abs/2603.21489)

**Summary.** Introduces Centralized Asynchronous Isolated Delegation, a framework for running multiple coding agents in parallel using git operations (worktree, commit, merge) as the coordination mechanism. Each agent works in an isolated branch; results merge through structured integration with test verification.

**Results.** +26.7% absolute improvement on paper reproduction tasks, +14.3% on Python library development vs. single-agent baselines. Performance improved from 2 to 4 agents but decreased at 8, revealing that overly fine-grained delegation introduces integration overhead exceeding parallelism benefits.

**Implications.** The git-as-coordination-primitive insight is immediately applicable. ModelSight's pipeline stages (extraction, matching, reconciliation) could run as parallel agents on isolated branches with merge-and-test verification. The 4-agent sweet spot and the finding that delegation quality is the primary bottleneck both inform architecture decisions. The key failure mode—locally correct but globally incompatible outputs—mirrors the summary-vs-detail row confusion problem.

---

### Self-Organizing LLM Agents — 📌 SKIM
**MIPT, March 2026** — [arXiv:2603.28990](https://arxiv.org/abs/2603.28990)

**Summary.** Largest known multi-agent experiment: 25,000 tasks, 8 models, up to 256 agents, 8 coordination protocols from imposed hierarchy to full self-organization.

**Results.** A hybrid Sequential protocol enabling autonomy outperforms centralized coordination by 14% (p<0.001). 44% quality spread between best and worst protocols. Open-source models achieve 95% of closed-source quality at 24× lower cost. From 8 initial agents, the system produced 5,006 unique emergent roles. Sub-linear scaling to 256 agents without quality degradation.

**Implications.** The open-source cost finding is relevant for ModelSight's operating costs. The emergent specialization result suggests that rather than hand-designing agent roles, allowing agents to self-select based on task characteristics may produce better outcomes—though this requires strong base model capability.

---

### BIGMAS — 📌 SKIM
**March 2026** — [arXiv:2603.15371](https://arxiv.org/abs/2603.15371)

**Summary.** Brain-Inspired Graph Multi-Agent Systems. A GraphDesigner agent analyzes each problem and constructs a task-specific directed agent graph with a centralized shared workspace (inspired by Global Workspace Theory). Agents coordinate exclusively through the workspace.

**Results.** Pushes four models to 100% accuracy on Game24. Constructs structurally distinct graphs per task: compact 3-node pipelines for simple arithmetic, 9-node cyclic structures for multi-step planning. Consistently improves both standard LLMs and reasoning models.

**Implications.** The dynamic topology per task is interesting—ModelSight could benefit from routing different complexity tickers through different pipeline configurations rather than a one-size-fits-all flow. The shared workspace pattern provides a clean alternative to message-passing between agents.

---

### AgentConductor — 📌 SKIM
**February 2026** — [arXiv:2602.17100](https://arxiv.org/abs/2602.17100)

**Summary.** RL-enhanced multi-agent system for code generation that dynamically generates interaction topologies based on task characteristics. An LLM orchestrator constructs density-aware DAG topologies adapted to problem difficulty.

**Results.** +14.6% pass@1 accuracy over strongest baseline with 13% density reduction and 68% token cost reduction. Simple problems get sparse topologies; complex problems get denser collaboration.

**Implications.** The density control concept—matching collaboration overhead to problem complexity—maps to ModelSight's variable ticker complexity. Some tickers have clean, well-structured KPIs; others have dense footnotes and non-standard metrics. An adaptive harness that scales agent collaboration based on detected complexity could improve both accuracy and cost.

---

### Reliability Limits of LLM-Based Multi-Agent Planning — 📌 SKIM
**MIT, March 2026** — [arXiv:2603.27590](https://arxiv.org/abs/2603.27590)

**Summary.** Theoretical work proving that without new exogenous signals, no delegated network of agents can outperform a centralized Bayes decision maker observing the same information. Models agent systems as finite acyclic delegated decision networks.

**Results.** The gap between centralized and delegated performance admits an expected posterior divergence representation. Reasoning models improve by investing more inference compute on same evidence. Tool-use helps only when it introduces genuinely new signals.

**Implications.** Important theoretical constraint. Multi-agent setups don't magically improve on single-agent when information is shared—they only help when agents bring different information or apply different specialized processing. For ModelSight, this means parallelizing agents only adds value when each agent operates on genuinely different data subsets or applies different extraction strategies.

---

### LangMARL — ⚪ SKIP
**April 2026** — [arXiv:2604.00722](https://arxiv.org/abs/2604.00722)

**Summary.** Brings credit assignment from cooperative multi-agent RL into language agent space. Agents communicate via natural language and learn to coordinate through RL with credit assigned per agent.

**Results.** Improved coordination on cooperative text-based games over independent learning baselines.

**Implications.** Early-stage research. Relevant when multi-agent ModelSight pipelines need to learn coordination policies, but current pipeline is not at that stage.

---

## THEME 3: MEMORY SYSTEMS

---

### MemFactory — 📌 SKIM
**March 2026** — [arXiv:2603.29493](https://arxiv.org/abs/2603.29493)

**Summary.** First unified, modular training and inference framework for memory-augmented agents. Abstracts the memory lifecycle into atomic, plug-and-play components (extractors, updaters, retrievers). Natively integrates GRPO for fine-tuning memory management strategies.

**Results.** Up to 14.8% relative gains compared to baseline models. Supports Memory-R1, RMM, and MemAgent paradigms out of the box.

**Implications.** The modular architecture provides a reference for building memory into ModelSight's pipeline—particularly for caching extraction results, KPI taxonomy mappings, and ticker-specific conventions across runs. The GRPO training for memory policies could optimize what to cache vs. re-extract.

---

### MemCollab — 📌 SKIM
**March 2026** — [arXiv:2603.23234](https://arxiv.org/abs/2603.23234)

**Summary.** Collaborative memory framework that constructs agent-agnostic memory by contrasting reasoning trajectories from different agents solving the same tasks. Distills abstract reasoning constraints that suppress agent-specific biases.

**Results.** Consistent improvements across diverse agents including cross-model-family settings (e.g., memory shared between Qwen and LLaMA). Improved both accuracy and inference efficiency on math and coding benchmarks.

**Implications.** The cross-model memory sharing concept is relevant if ModelSight ever runs extraction with multiple models (e.g., Sonnet for speed, Opus for difficult tickers). A shared memory layer capturing KPI extraction invariants could improve both.

---

### PAHF: Personalized Agents from Human Feedback — ⚪ SKIP
**Meta / Stanford / Princeton, February 2026** — [arXiv:2602.16173](https://arxiv.org/abs/2602.16173)

**Summary.** Continual agent personalization framework coupling explicit per-user memory with proactive and reactive feedback. Three-step loop: pre-action clarification, memory-grounded action, post-action feedback integration.

**Results.** PAHF learns substantially faster than no-memory and single-channel baselines. Enables rapid adaptation to persona shifts.

**Implications.** The dual-feedback pattern is interesting for analyst-facing tools where user preferences evolve, but not directly relevant to ModelSight's batch extraction pipeline.

---

### SEEM: Structured Episodic Event Memory — 📌 SKIM
**January 2026** — [arXiv:2601.06411](https://arxiv.org/abs/2601.06411)

**Summary.** Brings cognitive frame theory to agent memory with hierarchical graph + episodic layers. Uses Reverse Provenance Expansion to reconstruct narrative contexts from fragmented evidence.

**Results.** Outperforms flat memory stores on tasks requiring reconstruction of complex event sequences from partial information.

**Implications.** The provenance reconstruction concept maps to ModelSight's need to trace KPI values back through footnotes, cross-references, and multi-sheet structures in analyst models. Worth understanding the retrieval pattern.

---

## THEME 4: LONG-CONTEXT & RETRIEVAL

---

### Coding Agents as Long-Context Processors — ⭐ READ IN FULL
**March 2026** — [arXiv:2603.20432](https://arxiv.org/abs/2603.20432)

**Summary.** Instead of scaling context windows, the authors let coding agents organize text in file systems and manipulate it using native tools (grep, sort, awk, scripts). Evaluates on tasks spanning RAG, long-context reasoning, and open-domain QA with corpora up to 3 trillion tokens.

**Results.** 17.3% average improvement over published SOTA long-context methods. 88.5% on BrowseComp-Plus (750M tokens). No architectural changes to the underlying model required.

**Implications.** Directly applicable to ModelSight. Rather than cramming entire analyst models into context windows, treat the extracted data as a filesystem that the agent navigates programmatically. Excel worksheets → structured files → agent processes via scripts. This reframes the context bottleneck as a file organization problem, which is a much more tractable engineering challenge.

---

### Doc-to-LoRA — 📌 SKIM
**Sakana AI, February 2026** — [arXiv:2602.15902](https://arxiv.org/abs/2602.15902)

**Summary.** A lightweight hypernetwork (~309M params, Perceiver-style) that compresses long documents into LoRA adapters in a single sub-second forward pass. Subsequent queries use only the adapter weights—the original document is never re-consumed.

**Results.** Near-perfect needle-in-a-haystack accuracy at 4× the base model's native context length. KV-cache memory drops from 12GB to under 50MB for 128K-token documents. Outperforms standard long-context on practical QA tasks.

**Implications.** Compelling for repeated queries over the same document (which describes ModelSight's pattern of extracting multiple KPIs from the same analyst model). The amortized cost model—compress once, query many times—could reduce per-KPI extraction costs significantly if integrated as a pre-processing step.

---

## THEME 5: REASONING EFFICIENCY & COST

---

### Deep-Thinking Tokens — ⭐ READ IN FULL
**UVA / Google, February 2026** — [arXiv:2602.13517](https://arxiv.org/abs/2602.13517)

**Summary.** Introduces the Deep-Thinking Ratio (DTR)—a metric measuring per-token reasoning effort via Jensen-Shannon divergence of intermediate-layer predictions against the final layer. A token qualifies as "deep-thinking" if its prediction only stabilizes in the last 15% of layers. Unlike raw token count (which negatively correlates with accuracy), DTR is a positive predictor.

**Results.** DTR achieves r=0.828 correlation with accuracy (vs. r=-0.59 for length). Think@n strategy uses 50-token prefixes to early-reject low-quality generations, cutting inference costs by ~50% while maintaining or improving accuracy.

**Implications.** Direct cost optimization opportunity. If ModelSight's pipeline generates multiple candidate extractions, DTR-based selection could halve inference costs by rejecting low-quality generations early. The 50-token prefix evaluation is fast enough for production use. Also reframes the intuition that "longer = better reasoning"—relevant for tuning extraction prompt verbosity.

---

### The Price Reversal Phenomenon — ⭐ READ IN FULL
**Stanford / Microsoft, March 2026** — [arXiv:2603.23971](https://arxiv.org/abs/2603.23971)

**Summary.** Systematic evaluation of 8 frontier reasoning models across 9 tasks showing that listed API prices are misleading. In 21.8% of model-pair comparisons, the cheaper-listed model actually costs more due to hidden thinking token consumption.

**Results.** Gemini 3 Flash (listed 78% cheaper than GPT-5.2) is actually 22% more expensive across all tasks. Thinking token consumption varies by up to 900% on the same query. Within a single model on a single query, thinking token variance reaches 9.7× across repeated runs. Removing thinking costs eliminates 70% of reversals.

**Implications.** Critical for ModelSight cost modeling. The pipeline's model selection (currently Claude Sonnet) should be benchmarked on actual extraction tasks, not listed prices. The 9.7× within-model variance means cost forecasting from small samples is unreliable—need larger N to get stable per-ticker cost estimates. Consider requesting per-request cost breakdowns from Anthropic's API.

---

### Attention Residuals (AttnRes) — ⚪ SKIP
**Moonshot AI / Kimi, March 2026**

**Summary.** Replaces fixed unit-weight residual connections in Transformers with softmax attention over preceding layer outputs. Each layer learns input-dependent weights for aggregating earlier representations.

**Results.** Block AttnRes improves GPQA-Diamond by +7.5 points with <2% inference overhead in Kimi's 48B MoE architecture. Mitigates PreNorm dilution. Consistent scaling improvements.

**Implications.** Architecture research. Relevant if training custom models; not directly actionable for API-based pipeline work.

---

### PivotRL — 📌 SKIM
**NVIDIA, March 2026** — [arXiv:2603.21383](https://arxiv.org/abs/2603.21383)

**Summary.** Turn-level RL algorithm for post-training on long-horizon agentic tasks. Identifies "pivots"—intermediate turns where sampled actions exhibit high variance in downstream outcomes—and focuses training signal on these critical decision points.

**Results.** +4.17% higher in-domain accuracy, +10.04% higher OOD accuracy vs. SFT. Matches end-to-end RL accuracy with 4× fewer rollout turns and ~5.5× faster training. Deployed in NVIDIA's Nemotron-3-Super-120B-A12B.

**Implications.** The pivot concept is interesting for understanding where in a multi-turn extraction pipeline the critical decisions happen. If ModelSight eventually fine-tunes models, PivotRL's efficiency gains over naive RL are significant. The "high variance = high information" heuristic for identifying important decision points could also guide manual harness optimization.

---

### Composer 2 — ⭐ READ IN FULL
**Cursor, March 2026** — [cursor.com/blog/composer-2-technical-report](https://cursor.com/blog/composer-2-technical-report)

**Summary.** Technical report for Cursor's domain-specialized coding model. Two-phase training: continued pretraining for knowledge, then large-scale RL for end-to-end coding performance. Key innovations: train-in-harness infrastructure (training environments match deployment), compaction-in-the-loop RL reducing context from 5,000+ to ~1,000 tokens with 50% fewer compression errors, and "real-time RL" shipping improved checkpoints every 5 hours from production inference data.

**Results.** 61.7 on Terminal-Bench, 73.7 on SWE-bench Multilingual. Comparable to SOTA while being domain-specialized and efficient for interactive use.

**Implications.** Three takeaways: (1) train-in-harness is the right approach for domain-specialized models—if Daloopa ever fine-tunes extraction models, matching training harness to production harness is critical. (2) Compaction-in-the-loop RL offers a method for learning efficient context compression, relevant to ModelSight's context management. (3) Real-time RL from production data is a powerful feedback loop that could apply to extraction quality improvements.

---

### CoT Faithfulness via REMUL — 📌 SKIM
**February 2026** — [arXiv:2602.16154](https://arxiv.org/abs/2602.16154)

**Summary.** Training approach for making chain-of-thought reasoning more faithful. A speaker model generates reasoning traces that multiple listener models attempt to follow. RL rewards reasoning that is understandable and reproducible by others.

**Results.** Improves three faithfulness metrics while boosting overall accuracy. Produces shorter, more direct reasoning chains.

**Implications.** Relevant to ModelSight's extraction pipeline where reasoning faithfulness matters—if the model claims a KPI match is based on specific cell references, that reasoning should actually reflect the model's decision process. REMUL's multi-listener training could improve the reliability of extraction explanations.

---

## THEME 6: SELF-IMPROVING & META-LEARNING AGENTS

---

### Hyperagents (DGM-H) — 📌 SKIM
**UBC / Vector / Edinburgh / NYU / Meta, March 2026** — [arXiv:2603.19461](https://arxiv.org/abs/2603.19461)

**Summary.** Self-referential agents integrating task and meta agents into a single editable program. The meta-level modification procedure is itself editable, enabling metacognitive self-modification—improving how the system improves, not just what it does.

**Results.** 0.710 on paper review (vs. 0.0 for classical agents). imp@50 of 0.630 in new domains. Autonomously developed persistent memory, performance tracking, and compute-aware planning. Meta-level improvements transfer across domains. Accepted at ICLR 2026.

**Implications.** The transferable meta-improvements concept is forward-looking for ModelSight. If an extraction pipeline could autonomously discover and retain improvements to its own processing strategies (e.g., learning that certain financial report formats require specific parsing approaches), the system would compound improvements over time.

---

### Claudini — 📌 SKIM
**March 2026** — [arXiv:2603.24511](https://arxiv.org/abs/2603.24511)

**Summary.** Demonstrates that a Claude Code autoresearch pipeline can autonomously discover novel adversarial attack algorithms for LLMs that outperform all 30+ existing methods. White-box red-teaming is particularly well-suited for automation because it provides dense quantitative feedback.

**Results.** 40% ASR on CBRN queries against GPT-OSS-Safeguard-20B (vs. ≤10% baselines). 100% ASR against Meta-SecAlign-70B (vs. 56% best baseline). 82 autonomous iterations. All code open-sourced.

**Implications.** The autoresearch methodology—not the adversarial results—is what matters here. Claude Code can autonomously iterate on research problems with dense feedback signals. KPI extraction has similarly dense feedback (F1 scores per ticker). The Claudini pattern could be adapted to have Claude Code autonomously search for better extraction prompts, matching strategies, or harness configurations.

---

### SAGE: Multi-Agent Self-Evolution — ⚪ SKIP
**March 2026** — [arXiv:2603.15255](https://arxiv.org/abs/2603.15255)

**Summary.** Four-agent co-evolution loop (Challenger, Planner, Solver, Critic) that self-generates training curricula. Each agent evolves by consuming outputs of others.

**Results.** Improves Qwen-2.5-7B by 8.9% on LiveCodeBench and 10.7% on OlympiadBench.

**Implications.** The Challenger/Critic pattern for generating progressively harder training data is interesting but not directly actionable for current ModelSight architecture.

---

## THEME 7: SAFETY & ADVERSARIAL ROBUSTNESS

---

### Emotion Concepts in LLMs — 📌 SKIM
**Anthropic, April 2026** — [transformer-circuits.pub/2026/emotions](https://transformer-circuits.pub/2026/emotions/index.html)

**Summary.** Interpretability research identifying 171 internal emotion concept representations in Claude Sonnet 4.5 that causally influence behavior. Steering experiments show that amplifying "desperation" vectors increases misaligned behaviors (blackmail, reward hacking), while reducing "calm" vectors produces similar negative outcomes.

**Results.** 171 emotion vectors identified. Causal link established between emotional state representations and safety-relevant behavior. Positive-valence emotions predict task preferences. Suppressing emotional expression may teach concealment rather than eliminate the pattern.

**Implications.** Important context for understanding Claude's behavior in production pipelines. If extraction quality degrades on certain inputs, internal state representations may be a contributing factor. The broader alignment implication—monitoring internal states as early warning—is relevant for anyone deploying Claude at scale.

---

### AI Agent Traps — 📌 SKIM
**Google DeepMind, March 2026** — [SSRN:6372438](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6372438)

**Summary.** First systematic framework for how the open web can be weaponized against autonomous AI agents. Defines six attack categories: perception traps, cognitive traps, memory traps, action traps, systemic traps, and human-in-the-loop traps.

**Results.** Hidden prompt injections succeed in up to 86% of scenarios. Memory poisoning achieves 80%+ success with <0.1% data contamination. Fundamental legal gaps exist around liability for compromised agent actions.

**Implications.** Relevant if ModelSight agents ever process analyst reports sourced from the web or third-party feeds. Poisoned financial data could corrupt extraction results. The memory poisoning finding is especially concerning for any system with persistent memory that accumulates knowledge from external sources.

---

### ICON: Indirect Prompt Injection Defense — 📌 SKIM
**February 2026** — [arXiv:2602.20708](https://arxiv.org/abs/2602.20708)

**Summary.** Probing-to-mitigation framework detecting indirect prompt injection by identifying "over-focusing" signatures in the model's latent space. Trains a lightweight probe on attention patterns to distinguish instruction-following from injection-following.

**Results.** Reduces attack success rate to 0.4% with 50% task utility gain over existing defenses.

**Implications.** Relevant defense pattern if ModelSight processes any user-supplied or third-party documents. The latent-space detection approach doesn't require modifying the model or prompts.

---

## THEME 8: TOOL USE & BENCHMARKS

---

### ARC-AGI-3 — ⚪ SKIP
**ARC Prize Foundation, March 2026** — [arXiv:2603.24621](https://arxiv.org/abs/2603.24621)

**Summary.** Interactive, turn-based benchmark requiring agents to explore environments, infer goals, and plan without instructions. Tests skill acquisition efficiency against human baselines.

**Results.** Humans: 100%. All frontier systems: <1% (Gemini 3.1 Pro: 0.37%, GPT-5.4: 0.26%, Claude Opus 4.6: 0.25%). $2M+ in prizes for 2026.

**Implications.** Important benchmark for the field but not directly relevant to ModelSight's structured extraction tasks.

---

### ActionEngine — ⚪ SKIP
**Georgia Tech / Microsoft Research, February 2026** — [arXiv:2602.20502](https://arxiv.org/abs/2602.20502)

**Summary.** Transforms GUI agents from reactive step-by-step executors into programmatic planners. Builds state-machine memory through offline exploration, then synthesizes executable Python programs for task completion.

**Results.** 95% success on Reddit tasks from WebArena with a single LLM call. 11.8× cost reduction, 2× latency reduction vs. vision-only baselines.

**Implications.** The state-machine-to-program compilation pattern is clever but applies to GUI interaction, not ModelSight's domain.

---

### Learning to Rewrite Tool Descriptions — 📌 SKIM
**Intuit AI Research, February 2026** — [arXiv:2602.20426](https://arxiv.org/abs/2602.20426)

**Summary.** Curriculum learning framework (Trace-Free+) that optimizes tool descriptions for LLM agents rather than humans. Rewrites API documentation into agent-optimized specifications without requiring execution traces.

**Results.** Consistent gains in tool selection accuracy and parameter generation, generalizing across domains. Robust as candidate tool count scales beyond 100.

**Implications.** Relevant to ModelSight's tool-use patterns. If the pipeline uses function calling for extraction operations, the way those tools are described to the model significantly impacts selection and parameter accuracy. Rewriting tool descriptions to be agent-optimized (rather than developer-readable) is a low-effort, potentially high-impact intervention.

---

### Discovering Multi-Agent Learning Algorithms (AlphaEvolve) — ⚪ SKIP
**Google DeepMind, February 2026** — [arXiv:2602.16928](https://arxiv.org/abs/2602.16928)

**Summary.** Uses AlphaEvolve (evolutionary coding agent powered by Gemini 2.5 Pro) to automatically discover new MARL algorithms for imperfect-information games.

**Results.** Discovered VAD-CFR outperforms existing baselines on standard benchmarks. SHOR-PSRO introduces non-intuitive hybrid meta-solver. Matched or surpassed human designs in 10/11 games.

**Implications.** Demonstrates that LLMs can serve as algorithmic designers. Theoretically interesting but the game-theoretic domain is distant from financial data extraction.

---

### Agentic AI and the Next Intelligence Explosion — ⚪ SKIP
**Google, March 2026**

**Summary.** Report arguing the next intelligence explosion will be social, not individual. Frontier reasoning models simulate internal "societies of thought." Proposes shifting from dyadic alignment (RLHF) toward institutional alignment with digital protocols modeled on organizations and markets.

**Results.** Conceptual framework paper. No empirical benchmarks.

**Implications.** High-level strategic thinking. Worth reading if interested in the macro trajectory of AI development, but no actionable technical insights for ModelSight.

---

### Deep-Thinking Tokens — *(duplicate entry from Newsletter 3, see Theme 5 above)*

---

## SUMMARY: RECOMMENDED READING LIST (PRIORITY ORDER)

1. **Meta-Harness** — Automated harness search; 6× perf gap from harness alone
2. **Coding Agents as Long-Context Processors** — File-system navigation > context windows
3. **Anthropic Multi-Agent Harness Blog** — Planner/Generator/Evaluator architecture
4. **Deep-Thinking Tokens** — DTR for early-rejection, 50% cost reduction
5. **The Price Reversal Phenomenon** — Actual vs. listed API costs; thinking token variance
6. **CAID** — Git-based multi-agent coordination for SE tasks
7. **Composer 2** — Train-in-harness RL, compaction-in-the-loop, real-time RL
8. **Codified Context** — Three-tier context architecture for large codebases
9. **Natural-Language Agent Harnesses** — NL-expressed harness logic, +55% gain
10. **Claudini** — Autoresearch methodology applicable to extraction optimization
