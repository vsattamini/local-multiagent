#!/usr/bin/env python3
"""Generate all phase-3 experiment configs (ensemble, heterogeneous, random-router
control, greedy, large-N, MBPP+). New runs adopt the M1/M2 fixes
(decouple_decode_seed + clean_mode: strict)."""
import yaml
from pathlib import Path

MODELS = {"1.5b": "models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf",
          "3b": "models/qwen2.5-coder-3b-instruct-q4_k_m.gguf",
          "7b": "models/qwen2.5-coder-7b-instruct-q4_k_m.gguf"}
SYS = ("You are a Python coding assistant. Write clean, correct code that solves the "
       "problem. Only output the function implementation, no explanations or extra text.")
PERSONAS = [
    "You are a Python specialist in STRING manipulation, parsing, and text formatting.\nWrite clean, correct code. Only output the function implementation.",
    "You are a Python specialist in MATHEMATICS, arithmetic, and numeric algorithms.\nWrite clean, correct code. Only output the function implementation.",
    "You are a Python specialist in LIST and array data-structure operations.\nWrite clean, correct code. Only output the function implementation.",
    "You are a Python specialist in LOGIC, conditionals, and boolean validation.\nWrite clean, correct code. Only output the function implementation.",
]

def base(name, size, n_agents=3, router="affinity", rtemp=0.3, timeout=None, extra_agents=None,
         extra_exec=None, benchmark="humaneval", desc=""):
    tmo = timeout if timeout is not None else (10 if size == "7b" else 5)
    agents = {"n_agents": n_agents, "max_context_examples": 5}
    if extra_agents: agents.update(extra_agents)
    execu = {"timeout": tmo, "decouple_decode_seed": True, "clean_mode": "strict"}
    if extra_exec: execu.update(extra_exec)
    return {
        "experiment": {"name": name, "description": desc, "output_dir": f"results/{name}"},
        "model": {"path": MODELS[size], "context_length": 4096, "max_tokens": 512,
                  "generation_temperature": 0.2, "generation_top_p": 0.95},
        "agents": agents,
        "router": {"type": router, "temperature": rtemp},
        "tasks": {"n_tasks": 164, "task_types": ["string", "math", "list", "logic"],
                  "benchmark": benchmark},
        "execution": execu,
        "metrics": {"snapshot_interval": 10, "compute_significance": True, "n_permutations": 1000},
        "prompts": {"system": SYS},
        "random_seed": 42,
    }

def write(cfg):
    p = Path("config") / f"{cfg['experiment']['name']}.yaml"
    p.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))
    print("wrote", p)

cfgs = []
# 1) Ensemble (all agents per task) — 3B + 7B; type-filtered K=20, 4 agents
for size in ["3b", "7b"]:
    cfgs.append(base(f"ens_{size}", size, n_agents=4, router="round_robin", rtemp=0.3,
                     extra_agents={"max_context_examples": 20, "context_retrieval": "type_filtered",
                                   "context_show": 5, "ensemble_assignment": "all"},
                     desc=f"Ensemble: every agent attempts every task ({size}); enables GLMM + voting"))
# 2) Heterogeneous swarm: 2x1.5B + 1x3B, affinity
cfgs.append(base("het_swarm", "1.5b", n_agents=3, router="affinity", rtemp=0.3,
                 extra_agents={"models": ["1.5b", "1.5b", "3b"]},
                 desc="Heterogeneous swarm 2x1.5B+1x3B — competence-driven differentiation"))
# 3) RandomRouter control — 3 sizes (deconfounds S)
for size in ["1.5b", "3b", "7b"]:
    cfgs.append(base(f"rand_{size}", size, n_agents=3, router="random", rtemp=0.5,
                     desc=f"RandomRouter control ({size}) — S floor under non-affinity assignment"))
# 4) Greedy router — 3B
cfgs.append(base("greedy_3b", "3b", n_agents=3, router="greedy", rtemp=0.3,
                 desc="Greedy router (pure exploitation) — router-policy axis"))
# 5) Large-N population — 1.5B, n_agents 8/12/16, low temp + type-filtered
for n in [8, 12, 16]:
    cfgs.append(base(f"popN_{n}", "1.5b", n_agents=n, router="affinity", rtemp=0.1,
                     extra_agents={"max_context_examples": 20, "context_retrieval": "type_filtered", "context_show": 5},
                     desc=f"Large-N population n_agents={n} (1.5B) — robust S(N) curve"))
# 6) MBPP+ core: 3 sizes x {baseline T=0.5, low-temp T=0.1}
for size in ["1.5b", "3b", "7b"]:
    for tag, rt in [("baseline", 0.5), ("lowtemp", 0.1)]:
        cfgs.append(base(f"mbpp_{size}_{tag}", size, n_agents=3, router="affinity", rtemp=rt,
                         benchmark="mbpp", desc=f"MBPP+ {size} {tag} (router T={rt})"))

for c in cfgs:
    write(c)
print(f"\nTotal: {len(cfgs)} configs")
