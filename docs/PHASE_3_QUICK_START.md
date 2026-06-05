# Phase 3 Quick Start Card

**⚡ One-Page Reference for Next Agent**

---

## ✅ Status: Infrastructure Complete, Bug-Free

**Critical Bug Check**: ✅ NO test wrapping issues (verified in `src/swarm/executor.py` and `src/evaluation/swebench.py`)

---

## 🎯 Your Mission

Replace mock solution generation with real model inference, run experiments, analyze results.

---

## 🚀 Quick Commands

### Verify Setup
```bash
# Check Docker
docker ps

# Check dataset
cat data/swebench_curated.json | jq '.metadata'

# Check model (download if needed)
ls -lh models/qwen2.5-coder-7b-instruct-q4_k_m.gguf
```

### Test Run (3 issues, mock mode)
```bash
python scripts/run_swebench_experiment.py --condition experimental --n-issues 3
```

### Real Run (when model ready)
```bash
# Small test (5 issues, ~30 min)
python scripts/run_swebench_experiment.py \
    --condition experimental \
    --n-issues 5 \
    --model-path models/qwen2.5-coder-7b-instruct-q4_k_m.gguf

# Full experiment (50 issues × 3 conditions, 6-18 hours)
for cond in baseline control experimental; do
    python scripts/run_swebench_experiment.py \
        --condition $cond \
        --model-path models/qwen2.5-coder-7b-instruct-q4_k_m.gguf
done
```

### Analyze Results
```bash
python scripts/analyze_swebench_results.py
cat results/swebench/swebench_analysis.md
```

---

## 📁 Key Files

| File | What It Does |
|------|--------------|
| `data/swebench_curated.json` | 50 curated issues |
| `scripts/run_swebench_experiment.py` | Main runner (replace mock here) |
| `scripts/analyze_swebench_results.py` | Generate comparative report |
| `config/exp3_*.yaml` | 3 condition configs |
| `PHASE_3_HANDOVER_FINAL.md` | Complete handoff (read this!) |

---

## 🔧 What to Change

**File**: `scripts/run_swebench_experiment.py`

**Line 32-40** (current):
```python
def generate_mock_solution(issue: Dict, condition: str) -> str:
    return f"""# Mock solution for {issue['instance_id']}
# Condition: {condition}
def fix_issue():
    pass
"""
```

**Replace with**:
```python
def generate_real_solution(issue: Dict, condition: str, model) -> str:
    prompt = f"""Fix this issue:
Repository: {issue['repo']}
Problem: {issue['problem_statement']}

Generate a Python patch to fix it."""

    return model.generate(prompt, max_tokens=2048)
```

Then update line 66 in `run_experiment()`:
```python
# OLD: solution = generate_mock_solution(issue, condition)
# NEW: solution = generate_real_solution(issue, condition, model)
```

---

## 📊 Expected Results

### Realistic Expectations (SLMs on SWE-bench Lite)

| Condition | Resolve Rate | Interpretation |
|-----------|--------------|----------------|
| Baseline | 1-3% | Normal for 7B models |
| Control | 1-3% | Population size alone doesn't help |
| Experimental | 2-5% | If higher → specialization works! |

**Even small improvements (2% → 3%) are significant!**

---

## ⏱️ Time Estimates

| Scope | Without Tests | With Tests |
|-------|--------------|------------|
| 5 issues | 30 min | 2-4 hours |
| 10 issues | 1 hour | 4-8 hours |
| 50 issues (all) | 3-6 hours | 15-30 hours |
| 3 conditions × 50 | 9-18 hours | 45-90 hours |

**Recommendation**: Start with 5 issues to validate, then scale up.

---

## ⚠️ Common Issues & Fixes

### "Model not found"
```bash
# Download from Hugging Face
huggingface-cli download Qwen/Qwen2.5-Coder-7B-Instruct-GGUF \
    qwen2.5-coder-7b-instruct-q4_k_m.gguf \
    --local-dir models/
```

### "Docker not running"
```bash
sudo systemctl start docker
```

### "Out of disk space"
```bash
# Clean Docker images
docker system prune -af
```

### "CUDA out of memory"
```bash
# Run conditions sequentially (not parallel)
# Reduce n_agents in config files
```

---

## 📚 Documentation Hierarchy

1. **THIS FILE** - Quick start (you are here)
2. `PHASE_3_HANDOVER_FINAL.md` - Complete handoff
3. `docs/PHASE_3_SUMMARY.md` - Executive summary
4. `docs/PHASE_3_HANDOFF.md` - Technical details
5. `docs/plans/PHASE_3_EXECUTION.md` - Original plan

**Read in order**: 1 → 2 → 3

---

## ✅ Pre-Flight Checklist

- [ ] Docker running
- [ ] Model downloaded (~5GB)
- [ ] 50GB+ disk space free
- [ ] Read `PHASE_3_HANDOVER_FINAL.md`
- [ ] Tested mock run (3 issues)
- [ ] Understand expected time (9-18 hours for full run)

---

## 🎓 Research Value

**All outcomes are valuable**:
- ✅ Specialization helps → "Emergence transfers to realistic tasks"
- ✅ No improvement → "Characterizes limits of approach"
- ✅ Mixed results → "Domain-specific findings"

**You can't fail scientifically!** Just run, analyze, document.

---

## 🆘 If You Get Stuck

1. Check logs in `results/swebench/{condition}/`
2. Review inline documentation in Python files
3. Test with smaller subset (--n-issues 3)
4. Verify model loads: `python -c "from src.models.llama_cpp import LlamaCppModel; print('OK')"`

---

## 🎯 Success = Completion

**Your job is done when**:
1. All 3 conditions run (50 issues each)
2. `scripts/analyze_swebench_results.py` generates report
3. Results documented
4. Findings interpreted

**Time to completion**: 1-3 days (depending on compute)

---

**Good luck! The infrastructure is solid. Just plug in the model and run.** 🚀

---

*Quick Start Version: 1.0*
*Read `PHASE_3_HANDOVER_FINAL.md` for complete details*
