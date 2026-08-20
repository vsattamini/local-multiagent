# Decisive experiment slate (Review-2 ornery design agent, 2026-06-09)

## ⭐ #1 LOAD-BEARING — `xval_roles` (train/test held-out; the user's split idea, properly designed)
The ONLY leakage-free, co-observed test of "do roles form AND generalize."
- **Design:** affinity-route on TRAIN split (single-assignment → niches CAN form via selective
  exposure; S>0 measurable) → FREEZE agent contexts + router → TEST pass: every frozen agent
  attempts each HELD-OUT problem (ensemble → interaction computable) → test agent×type
  interaction on held-out, task-clustered outcomes.
- **Split:** 5-FOLD (not 80/20 — 80/20 leaves ~2 logic test items on HumanEval). Pool held-out
  predictions across folds → full per-type n, every prediction leakage-free. **On MBPP+** (logic
  n=31 viable; HumanEval logic n=10 too small → secondary only).
- **Conditions:** het [1.5,1.5,3B] + matched homogeneous [3B]×3 control (differ only in weights).
- **Analysis (pre-register before running):** GLMM success ~ agent*type + (1|task_id) on held-out;
  cluster on held-out task_id; report train-pass S vs random-router baseline + active-agent count.
- **Cost:** 5 folds × 3 seeds × 2 conditions ≈ **42–58 GPU-h**. VRAM ok (het 5.5GB; homog shares
  one 3B = 2.5GB).
- **Implement as ISOLATED standalone `scripts/run_xval_roles.py`** (do NOT edit experiment.py while
  the REST/extras pipeline is still running — re-import risk). Reuses agent/router/executor/loader.
- **Likely outcome (given v3 settled null):** het = uniform competence main effect that generalizes,
  NO held-out interaction; homog = neither → defensible leakage-free claim "context-only identical
  agents form no generalizing functional roles; heterogeneity = uniform competence + ensemble diversity."

## Free (0-GPU) hardeners — do on #1's output (and lock taxonomy BEFORE #1)
- **#4a label-free latent axis:** prompt-length / AST-size / complexity features instead of the 4
  hand labels (logic=fallback). Pre-register this axis BEFORE running #1. De-hostages the conclusion.
- **#6 held-out style classifier** (predict agent_id from held-out solution text) + **held-out power
  curve** + S_random baseline. Catches behavioral differentiation the success-rate LR misses.

## Conditional / scoped-out
- **#5 heterogeneity dose-response** [1.5,3B]/[1.5,7B]/[3,7B]: only deepen if #1 is positive; else a
  cheap plain-ensemble main-effect scaling figure. het_15_7b already queued in extras.
- **#2 in-sample affinity role-formation:** inferior to #1 (can't prove generalization); run only if #1's train-S is high.
- **#3 cooperative decomposition:** REJECT for this thesis (needs new orchestrator + real benchmark;
  only 2 SWE tasks available) → name as future work for *functional* (vs per-type) specialization.

## Throughline (methodology fix for the whole thesis)
Prove the null with **cluster-robust (GEE, cluster=task_id) interaction tests + power curves**, NOT
per-seed vote-counts. Naive "pool for power" gives a SPURIOUS p<0.0001 (pseudo-replication).
