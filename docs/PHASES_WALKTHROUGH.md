# Project Phases Walkthrough

This document provides an overview of the research project phases and current status.

---

## Phase Overview

```
Phase 1 (COMPLETE)          Phase 2 (READY)              Phase 3 (PLANNED)
┌────────────────────┐     ┌────────────────────┐     ┌────────────────────┐
│   PILOT STUDY      │     │  STATISTICAL       │     │   ECOLOGICAL       │
│                    │     │  VOLUME            │     │   VALIDATION       │
│ - 50 HumanEval     │ ──▶ │                    │ ──▶ │                    │
│ - 3 agents         │     │ - 164 HumanEval    │     │ - SWE-bench Lite   │
│ - S = 0.193 (GO)   │     │ - Multi-seed runs  │     │ - Real-world tasks │
│                    │     │ - Population sweep │     │ - Transfer test    │
└────────────────────┘     └────────────────────┘     └────────────────────┘
```

---

## Phase 1: Pilot Study ✅ COMPLETE

**Goal**: Validate infrastructure and detect initial emergence signal

**Results** (from `results/pilot_001/`):
- **Specialization Index (S)**: 0.193 → **GO** verdict (threshold: S > 0.1)
- **Infrastructure**: Working (model, routing, execution, logging)
- **Duration**: ~4 hours on local GPU

**Key Files**:
- Config: `config/pilot.yaml`
- Script: `scripts/run_pilot.py`
- Results: `results/pilot_001/`

---

## Phase 2: Statistical Volume 🔧 READY TO RUN

**Goal**: Establish statistical significance on full benchmark

### 2.1 Infrastructure (Complete)

| Component           | File                                  | Status      |
| ------------------- | ------------------------------------- | ----------- |
| Full categorization | `data/humaneval_categories_full.json` | ✅ 164 tasks |
| Multi-seed runner   | `scripts/run_experiment.py`           | ✅ Created   |
| Experiment configs  | `config/exp2.1_*.yaml`                | ✅ 4 configs |
| Statistical utils   | `src/swarm/significance_utils.py`     | ✅ Created   |

### 2.2 Experiments (Pending)

| Experiment                   | Config                     | Goal                   |
| ---------------------------- | -------------------------- | ---------------------- |
| Baseline (N=1)               | `exp2.1_baseline.yaml`     | Single agent baseline  |
| Control (N=3, random)        | `exp2.1_control.yaml`      | Random routing control |
| Experimental (N=3, affinity) | `exp2.1_experimental.yaml` | Pilot replication      |
| Scaling (N=5, affinity)      | `exp2.1_scaling.yaml`      | Population scaling     |

### 2.3 Commands to Run

```bash
# Run with multiple seeds for statistical significance
python scripts/run_experiment.py --config config/exp2.1_baseline.yaml --seeds 42 123 456
python scripts/run_experiment.py --config config/exp2.1_control.yaml --seeds 42 123 456
python scripts/run_experiment.py --config config/exp2.1_experimental.yaml --seeds 42 123 456
python scripts/run_experiment.py --config config/exp2.1_scaling.yaml --seeds 42 123 456

# Analyze results
python scripts/analyze_results.py results/exp2.1_experimental/
```

---

## Phase 3: Ecological Validation 📋 PLANNED

**Goal**: Test if emergent specialization transfers to realistic tasks

**Benchmark**: SWE-bench Lite (30-50 curated issues)

See: `docs/plans/PHASE_3_EXECUTION.md`

---

## File Structure

```
dissertacao/
├── config/
│   ├── pilot.yaml              # Phase 1 config
│   └── exp2.1_*.yaml           # Phase 2 configs
├── data/
│   └── humaneval_categories_full.json  # 164-task categorization
├── scripts/
│   ├── run_pilot.py            # Phase 1 runner
│   ├── run_experiment.py       # Phase 2 runner (multi-seed)
│   ├── categorize_tasks.py     # Task categorization
│   └── analyze_results.py      # Results analysis
├── src/swarm/
│   ├── experiment.py           # Core experiment orchestrator
│   ├── humaneval.py            # HumanEval loader
│   ├── metrics.py              # Emergence metrics
│   └── significance_utils.py   # Statistical tests
└── results/
    └── pilot_001/              # Phase 1 results
```

---

## Definition of Done

### Phase 2 Completion Criteria
- [ ] Full HumanEval (164 tasks) for N=1, N=3(Random), N=3(Affinity), N=5
- [ ] Statistical significance (p < 0.05) for Specialization Index
- [ ] Pass@1 comparison: Swarm vs Baseline
- [ ] Pareto frontier plot (Performance vs Tokens)

### Phase 3 Completion Criteria
- [ ] SWE-bench Lite setup and curated subset selection
- [ ] Relative performance comparison (Swarm vs Single model)
- [ ] Qualitative analysis of success/failure cases

---

*Last Updated: 2026-01-24*
