# Research lineage of "LLMs as General Pattern Machines"

**The 2023 paper by Mirchandani et al. sparked a prolific research ecosystem across abstract reasoning, robotics, and in-context learning theory, with its core insight — that LLMs function as general sequence modelers — now validated, challenged, and extended by over 170 citing works.** The most dramatic downstream result: ARC-AGI accuracy jumped from ~10% (the paper's era) to 87.5% (OpenAI o3, December 2024), driven by methods that build directly on the pattern-completion paradigm. Simultaneously, the robotics thread yielded production-ready action tokenizers (FAST, π0-FAST) and zero-shot robot controllers, while ICL theory matured to explain *why* transformers function as pattern machines — they implement mesa-optimization and approximate Bayesian inference during their forward pass.

---

## Analysis of the four candidate papers

### Paper 1: REASON (arXiv:2601.20784) — tangentially related hardware work

**Title:** "REASON: Accelerating Probabilistic Logical Reasoning for Scalable Neuro-Symbolic Intelligence"
**Authors:** Zishen Wan, Che-Kai Liu, Jiayi Qian, Hanchen Yang, Arijit Raychowdhury, Tushar Krishna (Georgia Tech)
**Date/Venue:** January 2026; HPCA 2026 Main Conference (Sydney, Australia)

This paper addresses computational bottlenecks in neuro-symbolic AI by proposing a three-level hardware co-design framework — algorithm-level DAG representations, a reconfigurable tree-based processing fabric, and GPU-integrated pipelines. It achieves **12–50× speedup** and **310–681× energy efficiency** over desktop and edge GPUs across six neuro-symbolic workloads. The connection to "LLMs as General Pattern Machines" is **indirect and tangential**. Both papers address AI reasoning, but from opposing paradigms: Mirchandani et al. demonstrate implicit pattern-based reasoning via LLM sequence completion, while REASON accelerates explicit symbolic and probabilistic reasoning engines. There is no direct citation relationship.

### Paper 2: arXiv:2603.20910 — not found

Despite extensive searching across arXiv, Google Scholar, and Semantic Scholar, **no paper with this arXiv ID could be located**. Nearby IDs in the 2603.208xx–2603.209xx range exist (confirming the numbering range is active for March 2026 submissions), but 2603.20910 itself returns no results. The paper may not yet have been published, may have been withdrawn, or the ID may contain a typographical error.

### Paper 3: ProQuest dissertation — inaccessible

The ProQuest URL (openview/af17953041114ad9722bb8bd925e6e98) could not be accessed due to authentication barriers. The dissertation's metadata did not appear in any public index (Google Scholar, Semantic Scholar, or web search results). Institutional ProQuest access would be required to retrieve this document. The `cbl=18750` and `diss=y` parameters confirm it is a doctoral dissertation in ProQuest's database.

### Paper 4: "Hallucinate or Memorize?" (arXiv:2511.08877) — thematically adjacent

**Title:** "Hallucinate or Memorize? The Two Sides of Probabilistic Learning in Large Language Models"
**Authors:** Junichiro Niimi (Meijo University; RIKEN AIP)
**Date/Venue:** November 2025; arXiv preprint

This paper investigates LLM hallucination in bibliographic recommendation, finding that citation count serves as a proxy for training data redundancy. Papers cited more than **~1,000 times** cross a threshold into near-verbatim memorization (cosine similarity >0.95), while lower-citation papers trigger probabilistic generation that produces hybrid or contaminated outputs. The relationship to Pattern Machines is **mechanistic but inverted**: Mirchandani et al. demonstrate the generalization capability of LLMs as pattern completers, while Niimi characterizes the boundary where generation shifts from flexible pattern completion to rigid memorization. This probes the limits of the "general pattern machine" paradigm — pattern completion works well for structural and syntactic patterns but degrades for specific factual content unless heavily memorized.

---

## From 10% to 87%: the ARC reasoning explosion

The most spectacular downstream impact of the Pattern Machines paper has been on the ARC-AGI benchmark. Mirchandani et al. first demonstrated that LLMs could complete ARC spatial patterns when prompted as ASCII art. By 2025, this insight had catalyzed a competitive ecosystem that dramatically advanced the state of the art.

**Ryan Greenblatt** (Redwood Research, June 2024) achieved ~42–50% on ARC-AGI by having GPT-4o generate ~8,000 Python programs per task, using multiple ASCII/grid representations and iterative debugging. Performance scaled log-linearly with sampled programs — a direct operationalization of using LLMs' pattern understanding to *write programs* rather than directly complete patterns. The **test-time training** paradigm emerged simultaneously: Akyürek et al. (MIT, November 2024) showed that temporarily updating an 8B model's parameters on a task's in-context examples yielded up to **6× improvement** over fine-tuned baselines, reaching 53% on ARC public eval and **61.9%** when ensembled with program synthesis. Li et al. (Cornell, November 2024; ICLR 2025 Best Paper, ARC Prize 2024) formalized the distinction between *induction* (inferring Python programs) and *transduction* (directly predicting outputs), showing they solve complementary task types — induction excels at precise computation, transduction at fuzzier perceptual concepts.

**OpenAI's o3** scored **75.7%** on ARC-AGI semi-private eval (and 87.5% at high compute) in December 2024, a step-function increase from GPT-4o's 5%. François Chollet called it "a surprising and important step-function increase in AI capabilities." The subsequent ARC-AGI-2 benchmark (launched 2025) proved much harder, with top scores reaching only **24%**, indicating that the pattern-machine paradigm still has fundamental limitations. Notably, the **Tiny Recursive Model** by Jolicoeur-Martineau (Samsung AI, ARC Prize 2025 Paper Award) achieved 45% on ARC-AGI-1 with only **~7 million parameters** — less than 0.01% of frontier LLMs — suggesting that recursive refinement may matter more than raw scale for abstract pattern completion.

---

## Token-invariance meets non-linguistic domains

Mirchandani et al.'s striking finding that pattern completion persists even with randomly remapped tokens sparked investigation into LLMs as modality-agnostic sequence processors. **Time-LLM** (Jin et al., ICLR 2024) explicitly cited Pattern Machines and demonstrated that frozen LLMs can forecast time series by "reprogramming" numerical patches into text prototype representations, outperforming specialized forecasting models in zero-shot scenarios. Tang et al. (2024) provided a finer-grained analysis, finding that LLMs excel at predicting series with clear periodicity but struggle with aperiodic data — mapping the boundary conditions of LLMs as numeric pattern completers.

The robustness of token-invariant reasoning has been both supported and challenged. **Webb, Holyoak, and Lu** (April 2024) defended emergent analogical reasoning in LLMs, showing GPT-4 with code execution can generalize to counterfactual analogy tasks. **Lewis and Mitchell** (Bristol/Santa Fe Institute, February 2024) countered with evidence that LLM accuracy drops sharply on permuted-alphabet letter-string analogies while human performance remains stable — directly challenging the depth of token-invariance reported by Mirchandani et al. **Xu et al.** (TMLR 2024) found that object-based text representations dramatically improve GPT-4's ARC performance, revealing that *how* patterns are tokenized matters enormously even if the specific tokens do not.

More recent work has pushed toward explicitly modality-agnostic architectures. **Mull-Tokens** (Ray et al., December 2025) introduced latent tokens that hold intermediate reasoning information in either image or text modalities, enabling free-form reasoning that transcends any single modality. **NumericBench** (ACL 2025) provided systematic evaluation of LLMs on fundamental numerical abilities, revealing significant gaps in numeric sequence prediction, contextual retrieval, and arithmetic — precisely testing the "general sequence modeler" claim at scale.

---

## Robots that complete patterns: from concept to deployment

The robotics thread of Pattern Machines has seen the most direct path from concept to practical systems. The original paper demonstrated motion extrapolation and reward-conditioned trajectory completion on CartPole; by 2025, multiple groups had scaled these ideas to real manipulation and locomotion.

**Kwon, Di Palo, and Johns** (Imperial College, RA-L 2024) showed GPT-4 can predict dense end-effector pose sequences for manipulation **zero-shot** — without in-context examples, motion primitives, or optimizers — across 30 real-world tasks. **Wang et al.** (UC Berkeley, CDC 2024) demonstrated LLMs as 10 Hz feedback controllers for quadruped locomotion, generating joint-position targets from few-shot observation-action prompts. **RoboPrompt** (Yin et al., UC Berkeley, ICRA 2025) achieved **51.8% success** across 16 RLBench tasks using pure text-only LLM in-context learning for 6-DoF action prediction — the most direct extension of Pattern Machines' insight to multi-task manipulation. **Carvalho and Nolfi** (June 2025) explicitly built on Mirchandani et al., adding iterative refinement loops where LLMs generate actions from sensory-motor data alone and improve through reward-conditioned prompt updates.

The critical infrastructure enabling this progress has been **action tokenization**. **FAST** (Pertsch et al., January 2025), developed partly by Pattern Machines co-authors Brian Ichter and Danny Driess, introduced frequency-space action tokenization using DCT compression. FAST solved the practical barrier Mirchandani et al. identified — naïve per-dimension binning fails for dexterous control — enabling the **π0-FAST** system to match diffusion-based vision-language-action models at **5× faster training**. **Moto** (Chen et al., ICCV 2025) learned latent motion tokens from video via VQ-VAE, pre-training a GPT-style model on motion token sequences for robot control. **ARP** (Zhang et al., RA-L 2025) built a full autoregressive action-sequence architecture that matches or beats environment-specific baselines across Push-T, ALOHA, and RLBench. Several original Pattern Machines co-authors (Xia, Zeng, Sadigh) continued extending LLM-robotics integration through **generative expressive behaviors** (HRI 2024) and **RoboVQA** (Sermanet, Mirchandani et al., 2023).

---

## Why transformers are pattern machines: theoretical foundations

The theoretical understanding of *why* LLMs function as general pattern completers matured significantly in 2024–2025, converging on two complementary explanations: **mesa-optimization** and **Bayesian inference**.

**Von Oswald et al.** (ICLR 2024) reverse-engineered autoregressive transformers trained on sequence prediction, discovering that standard next-token prediction gives rise to gradient-based **mesa-optimization** — the model implicitly constructs internal objectives and optimizes them during the forward pass. **Jin et al.** (NeurIPS 2024) provided formal proof that autoregressively trained transformers learn to implement one step of gradient descent to minimize an OLS problem in-context. This explains how LLMs can adapt to novel patterns at inference time without parameter updates — the forward pass *is* a learning algorithm.

The Bayesian perspective offers a complementary account. **Panwar, Ahuja, and Goyal** (ICLR 2024) showed empirically that high-capacity transformers mimic the Bayesian predictor across diverse function families, explaining how transformers handle task mixtures as a natural consequence of posterior inference over the pretraining distribution. **Reuter et al.** (ICML 2025) extended this to full Bayesian posterior inference, demonstrating that transformers can perform complete statistical inference in-context, matching MCMC and variational methods. This explains the token-remapping robustness Mirchandani et al. observed: if ICL approximates Bayesian inference over abstract program structures, surface token identity becomes irrelevant.

At the circuit level, **Todd et al.** (ICLR 2024) discovered **function vectors** — compact representations in attention heads that encode demonstrated tasks during ICL. These vectors can be extracted from a few heads and transplanted to trigger task execution in unrelated contexts. **Yin and Steinhardt** (ICML 2025) showed that function-vector heads, not simple induction heads, primarily drive ICL in larger models, with FV heads often starting as induction heads during training before transitioning to more complex mechanisms. **Bhattamishra et al.** (ICLR 2024, Oral) demonstrated that transformers can adaptively select between distinct algorithms for a single task depending on in-context examples — even for discrete Boolean function classes guaranteed absent from training data.

---

## Program synthesis as pattern completion's industrial form

The program synthesis thread represents the most practically successful operationalization of the pattern-machine insight. Rather than having LLMs directly complete output patterns, this approach leverages their pattern understanding to *generate transformation programs*.

**Hypothesis Search** (Wang et al., Stanford, ICLR 2024) pioneered LLM-driven inductive reasoning by generating hypotheses at multiple abstraction levels — natural-language descriptions, then Python implementations verified on examples. On 100 ARC problems, the automated pipeline achieved **30% accuracy** vs. 17% for direct prompting. **SOAR** (Pourcel et al., ICML 2025) embedded LLM program generation in a self-improving evolutionary loop: the model alternates between evolutionary search over candidate programs and hindsight fine-tuning on its own search traces, reaching **52% on ARC-AGI-1** without human-engineered DSLs. **TransCoder** (Bednarek and Krawiec, NeSy 2024) took the opposite architectural approach, using a neurosymbolic system that synthesizes programs in a typed DSL with a "learning from mistakes" paradigm.

The **ARC Prize 2024 Technical Report** (Chollet, Knoop et al., December 2024) documented that all top-performing approaches combined program synthesis with direct prediction. The report identified three breakthrough paradigm categories: deep learning-guided program synthesis, test-time training for transduction, and ensembles of both. State-of-the-art rose from **33% to 55.5%** on the private evaluation set, with the critical insight being that induction and transduction solve different task types. **From Reasoning to Generalization** (Lei et al., May 2025) augmented LLM program generation with structured knowledge priors organized in a hierarchical ontology (KAAR), achieving **~5% absolute gains** and up to 64.5% relative improvement over baseline program synthesis.

---

## The research landscape at a glance

| Research thread | Key milestone | Representative papers |
|---|---|---|
| ARC abstract reasoning | 10% → 87.5% accuracy (2023–2024) | Greenblatt 2024; Akyürek et al. 2024; o3 (Dec 2024) |
| Token-invariant modeling | Frozen LLMs forecast time series | Time-LLM (ICLR 2024); Mull-Tokens (2025) |
| Robotics via sequence completion | Production action tokenizers | FAST/π0-FAST (2025); RoboPrompt (ICRA 2025) |
| ICL theory | Mesa-optimization + Bayesian accounts | Von Oswald et al. (ICLR 2024); Panwar et al. (ICLR 2024) |
| Program synthesis for ARC | Self-improving evolutionary search | SOAR (ICML 2025); Li et al. (ICLR 2025) |

---

## Convergence and open frontiers

The research lineage of "LLMs as General Pattern Machines" has crystallized around a core insight that has proven remarkably generative: **LLMs' value as reasoning engines derives from their implicit implementation of learning algorithms during the forward pass, not merely from memorized knowledge.** This explains both the successes (token-invariant pattern completion, zero-shot robot control, ARC solving) and the failures (degradation on counterfactual analogies, brittleness on ARC-AGI-2, numeric reasoning gaps).

Three unresolved tensions define the frontier. First, the **scale vs. architecture debate**: Jolicoeur-Martineau's 7M-parameter Tiny Recursive Model matching or beating billion-parameter LLMs on ARC suggests that recursive refinement may be more fundamental than pretraining scale for abstract reasoning. Second, the **induction-transduction complementarity** formalized by Li et al. implies that no single paradigm — whether direct pattern completion or program synthesis — suffices for general abstract reasoning. Third, the **tokenization bottleneck** in robotics remains partially unsolved: FAST and Moto represent significant advances, but the gap between language-native and action-native tokens continues to constrain real-time deployment. The original paper's central question — whether LLMs are truly general pattern machines or sophisticated pattern matchers limited to their training distribution — remains open, with evidence accumulating on both sides.