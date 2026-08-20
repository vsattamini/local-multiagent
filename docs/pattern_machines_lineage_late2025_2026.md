# Research Lineage: "LLMs as General Pattern Machines" (Mirchandani et al., CoRL 2023)

## Late 2025–2026 Developments

The research pace accelerated dramatically in late 2025 through early 2026. This addendum covers work the initial lineage document missed by over-indexing on 2024.

---

### ARC Reasoning: Saturation-and-Reset Cycle

---

#### The ARC of Progress towards AGI: A Living Survey
**Vahdati et al., March 2026** — [arXiv:2603.13372](https://arxiv.org/abs/2603.13372)

**Summary.** First cross-generation analysis of 82 approaches across three ARC benchmark versions and two competition years. Definitive snapshot of the field as of February 2026.

**Results.** ARC-AGI-1: systems now reach 93.0% (Opus 4.6) — up from ~10% when Pattern Machines was published. ARC-AGI-2: 68.8%. ARC-AGI-3: 13%. Humans: near-perfect across all versions. Consistent 2–3× degradation across all paradigms (program synthesis, neuro-symbolic, neural) from one version to the next. Cost fell 390× in one year (o3's $4,500/task → GPT-5.2's $12/task). Small-scale entries (660M–8B params) achieve competitive results.

**Connection to Pattern Machines.** Validates the original paper's hypothesis that LLMs have latent pattern manipulation abilities, while also demonstrating hard limits. The cross-paradigm consistency of degradation across versions suggests a shared fundamental limitation in compositional generalization — not an artifact of any single approach. The finding that efficiency matters more than scale directly supports Pattern Machines' token-invariance result, where reasoning persisted with arbitrary vocabularies.

---

#### ARC-AGI-2
**Chollet, Knoop, Kamradt, Landers, Pinkard; May 2025 / revised January 2026** — [arXiv:2505.11831](https://arxiv.org/abs/2505.11831)

**Summary.** Upgraded version preserving input-output pair format with harder tasks measuring higher fluid intelligence levels. Extensive human testing baselines included.

**Results.** Top ARC Prize 2025 winners needed hundreds of thousands of synthetic examples to reach 24%. Human accuracy remains comparable to ARC-AGI-1 levels. Confirms that current AI reasoning is knowledge-bound.

**Connection to Pattern Machines.** The knowledge-boundedness finding challenges the "general pattern machine" framing. LLMs can complete patterns they've been exposed to variants of, but generating genuinely novel transformations from limited examples remains hard. This is the "breadth of generalization" question Pattern Machines raised but couldn't definitively answer.

---

#### ARC-AGI-3
**ARC Prize Foundation, March 2026** — [arXiv:2603.24621](https://arxiv.org/abs/2603.24621)

**Summary.** Fundamental format change: turn-based interactive environments requiring exploration, planning, memory, goal acquisition, and alignment. First interactive agentic benchmark for abstract reasoning.

**Results.** Humans: 100%. All frontier systems: <1%. Gemini 3.1 Pro: 0.37%. GPT-5.4: 0.26%. Claude Opus 4.6: 0.25%. Efficiency-based scoring caps AI at 5× human actions per level.

**Connection to Pattern Machines.** Moves entirely beyond static pattern completion. Pattern Machines showed LLMs can complete patterns from examples; ARC-AGI-3 asks whether they can discover patterns through interaction. The near-zero scores suggest that interactive skill acquisition is a fundamentally different capability from few-shot pattern matching.

---

#### Product of Experts with LLMs (ARC-AGI)
**Franzen et al., December 2025** — [arXiv:2505.07859](https://arxiv.org/abs/2505.07859)

**Summary.** Open-source SOTA on ARC-AGI-1 using task-specific data augmentations, DFS over LLM predictions, and the LLM as both generator and scorer via output probabilities. Product-of-experts scoring across augmented perspectives.

**Results.** 71.6% on ARC-AGI-1 public eval (286.5/400 solved). $0.02/task (vs. o3's $17/task). Surpasses average human performance. Transparent and reproducible.

**Connection to Pattern Machines.** Directly operationalizes Pattern Machines' insight. Rather than asking the LLM to complete patterns once, this method amplifies the model's latent pattern understanding through search over multiple representations — the same pattern viewed from different augmented perspectives. The product-of-experts scoring is a principled formalization of what Pattern Machines showed informally: LLMs carry more pattern knowledge than single forward passes reveal.

---

#### ArcMemo: Abstract Reasoning Composition with Lifelong Memory
**Ho et al., October 2025** — [arXiv:2509.04439](https://arxiv.org/abs/2509.04439)

**Summary.** Introduces lifelong concept memory for ARC reasoning. The model accumulates abstract takeaways from previous solve attempts and retrieves relevant memories for new tasks.

**Results.** 7.5% relative gain over strong no-memory baseline (55.17 → 59.33). Abstract concept memories outperform concrete memories at all inference scales. Dynamic updates during test-time outperform fixed settings, supporting self-improvement hypothesis.

**Connection to Pattern Machines.** Addresses a key limitation of the original paper's setup: each pattern was completed independently. ArcMemo shows that cross-task pattern knowledge — learning which types of transformations tend to work — accumulates and transfers. This is the "meta-pattern" level that Pattern Machines hinted at with PCFG composition experiments.

---

### ICL Theory: From Empirical to Information-Theoretic

---

#### Next-token pretraining implies in-context learning
**Riechers, Bigelow, Alt, Shai (Simplex/Astera Institute); May 2025** — [arXiv:2505.18373](https://arxiv.org/abs/2505.18373)

**Summary.** Proves that ICL is a mathematically inevitable consequence of successful next-token prediction loss minimization — not an exotic emergent property. Provides an information-theoretic framework that precisely predicts ICL dynamics.

**Results.** Framework reproduces phase transitions in induction head formation and power-law scaling of in-context loss on synthetic datasets. Shows a model's ICL performance on any task is mathematically coupled to the ensemble of tasks seen in pretraining. The result is architecture- and modality-independent.

**Connection to Pattern Machines.** This is the strongest theoretical vindication of Pattern Machines' central observation. If ICL inevitably arises from next-token prediction — regardless of architecture or modality — then token-invariant pattern completion is not surprising but *expected*. The architecture-independence explains why Pattern Machines' results held across different LLM families and with random token remapping. The theory grounds the "why" that the original paper could only hypothesize about.

---

#### Neural networks leverage nominally quantum and post-quantum representations
**Riechers, Elliott, Shai; July 2025** — [arXiv:2507.07432](https://arxiv.org/abs/2507.07432)

**Summary.** Shows that transformers pretrained on next-token prediction intrinsically discover and represent beliefs over low-dimensional generative models of their training data, performing iterative Bayesian updates during inference.

**Results.** Demonstrates that neural networks transcend limits of classical computational models by leveraging continuous activation spaces. Small networks can represent post-quantum belief geometries with few neurons. GPTs learn Generalized Probabilistic Theories (GPTs) — "GPTs learn GPTs."

**Connection to Pattern Machines.** Provides the mechanistic explanation for token invariance. If the model learns compressed *world models* (not surface token statistics), then token identity is irrelevant — what matters is the structural relationships. This is exactly what Pattern Machines observed empirically with random token remapping but couldn't explain.

---

#### Let's (not) just put things in Context: Test-Time Training for Long-Context LLMs
**Bansal et al., December 2025** — [arXiv:2512.13898](https://arxiv.org/abs/2512.13898)

**Summary.** Identifies score dilution in static self-attention — thinking tokens show diminishing returns at long context. Proposes targeted gradient updates on given context at test time.

**Results.** Inference-time thinking strategies fail at long context due to attention dilution. Test-time gradient updates provably overcome limitations of static self-attention. Extends TTT paradigm (which reached 61.9% on ARC) to general long-context tasks.

**Connection to Pattern Machines.** Suggests that static pattern completion (the Pattern Machines paradigm) has inherent length limitations due to attention mechanics. Dynamic model adaptation — temporarily updating weights based on the pattern being completed — may be necessary for complex or large-scale patterns. This connects directly to the ARC test-time training results that dominated the 2024–2025 competitions.

---

### VLA Models: Robotics-as-Sequence-Completion Goes Industrial

---

#### FAST: Efficient Action Tokenization for Vision-Language-Action Models
**Pertsch et al. (including Pattern Machines co-authors Ichter, Driess); January 2025** — [arXiv:2501.09747](https://arxiv.org/abs/2501.09747)

**Summary.** Introduces frequency-space action tokenization using DCT compression and BPE for continuous robot action sequences. Solves the action tokenization bottleneck identified in Pattern Machines.

**Results.** Enables VLA models to handle dexterous, high-frequency tasks where naive per-dimension binning fails completely. π0-FAST matches diffusion-based VLA models at 5× faster training.

**Connection to Pattern Machines.** This is the direct continuation by original Pattern Machines authors. The 2023 paper showed LLMs could complete simple numeric trajectory sequences; FAST solves the engineering problem of making this work for real robot control at production quality. The DCT approach elegantly addresses the tokenization challenge that constrained the original paper's robotics experiments.

---

#### ReMem-VLA: Recurrent Memory for Vision-Language-Action Models
**March 2026** — [arXiv:2603.12942](https://arxiv.org/abs/2603.12942)

**Summary.** Adds dual-level recurrent memory to VLA models. Frame-level queries for short-term retention, chunk-level queries for long-term context. Past observation prediction as auxiliary training objective.

**Results.** Significantly outperforms memory-free VLA baselines across spatial, temporal, episodic, sequential, and visual memory benchmarks in both simulation and real-world tasks.

**Connection to Pattern Machines.** Addresses Pattern Machines' Markovian limitation. The original paper treated each pattern completion as independent; ReMem-VLA shows that maintaining temporal context across sequential actions is essential for complex robot tasks. The model must remember what patterns it has already executed to know what comes next — going beyond single-shot completion to ongoing sequence management.

---

#### VLA Survey Landscape (Late 2025 – Early 2026)
Multiple comprehensive surveys document the field's maturation:

- **Pure VLA Models Survey** (September 2025, [arXiv:2509.19012](https://arxiv.org/abs/2509.19012)) — Taxonomy of autoregressive, diffusion, RL, hybrid, and specialized VLA paradigms
- **Efficient VLA Survey** (October 2025, [arXiv:2510.17111](https://arxiv.org/abs/2510.17111)) — Focus on token compression, action chunking, FAST-style DCT tokenization
- **Large VLM-based VLA Survey** (August 2025, [arXiv:2508.13073](https://arxiv.org/abs/2508.13073)) — Single-system vs. dual-system (System 1/System 2) architectures
- **VLA Concepts, Progress, Applications** (March 2026 revision, [arXiv:2505.04769](https://arxiv.org/abs/2505.04769)) — Includes humanoid applications, the "sentence is a trajectory" framing

**Connection to Pattern Machines.** The explosion of VLA surveys confirms that Pattern Machines' core insight — treating robot control as autoregressive sequence prediction — has become the dominant paradigm. The field now debates optimization details (autoregressive vs. diffusion decoding, single vs. dual system) rather than whether the sequence-completion framing is valid.

---

#### Convergence on Future-State Prediction (March 2026 Robotics Digest)
Four independent papers arrive at "dream ahead" as a structural principle:
- **DIAL** — latent intent bottleneck forcing predicted visual future before action (SOTA on RoboCasa GR1, 10× fewer demos)
- **CLaD** — cross-modal latent foresight grounding diffusion policy (94.7% on LIBERO-LONG)
- **LatentPilot** — latent tokens as compact world model across timesteps (SOTA on R2R-CE, RxR-CE)
- **RAAP** — retrieval-augmented decoupling of contact localization and action direction

**Connection to Pattern Machines.** Pattern Machines showed LLMs can extrapolate state sequences; these papers show that *predicting the future state before acting* is the key architectural pattern. The model completes the pattern of "what will the world look like" before generating "what action to take" — a two-stage pattern completion that outperforms direct action prediction.

---

### Context Management

---

#### Recursive Language Models
**Prime Intellect, March 2026** — [Blog](https://www.primeintellect.ai/blog/rlm); Paper at [arXiv:2512.24601](https://arxiv.org/abs/2512.24601)

**Summary.** Models that actively manage their own context by delegating to Python scripts and sub-LLMs rather than summarizing. Context folding without information loss. Trained end-to-end with RL.

**Results.** Proposed as the next major breakthrough for long-horizon agent tasks. Currently a research focus at Prime Intellect, with early results showing effective context management across extended reasoning sessions.

**Connection to Pattern Machines.** Extends the pattern machine concept to self-organizing systems. Rather than completing a pattern within a fixed context, the RLM decides what patterns to hold in context and what to delegate — a meta-level pattern completion over context management strategies themselves.

---

## Summary: What Changed from the Initial Lineage Document

The initial lineage (covering primarily 2024) underrepresented several major developments:

1. **ICL Theory matured dramatically in mid-2025.** Riechers et al. proved that ICL — the mechanism underlying all pattern completion — is a mathematical inevitability of NTP training, not an emergent property. This is the most important theoretical result for understanding why Pattern Machines work.

2. **ARC reasoning hit both new highs and new walls.** 93% on ARC-AGI-1 demonstrates the pattern completion paradigm's power. Sub-1% on ARC-AGI-3 demonstrates its limits. The transition from static to interactive benchmarks marks the boundary of the "pattern machine" framing.

3. **VLA became a named subfield.** Five major surveys in 6 months, standardized architectures, production deployments. The Pattern Machines proof-of-concept (CartPole trajectory completion) has become the dominant paradigm for robot control.

4. **Memory and self-improvement enter the picture.** ArcMemo, ReMem-VLA, and context folding all address the same limitation: single-shot pattern completion is insufficient for complex tasks. The next generation requires pattern machines that learn from their own pattern completions.

5. **Test-time adaptation is the key breakthrough mechanism.** From ARC-AGI competitions to long-context TTT, the most impactful results come from temporarily adapting the model to the specific pattern at hand — going beyond static in-context learning to dynamic weight updates.
