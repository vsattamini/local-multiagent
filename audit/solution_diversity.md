# GAP 2 — Unmeasured differentiation: diversity, voting, style

Ensemble ens_3b (4 agents x 164 tasks x 9 seeds). EXPLORATORY.

## A. Voting benefit (diversity payoff)

- mean per-agent pass@1: 0.822 ± 0.035
- any@N (≥1 of 4 correct): 0.885 ± 0.025
- majority@N (strict, >½ of 4; 2–2 ties count as fail): 0.805 ± 0.039
- **any@N − pass@1 = +0.063** [95% CI +0.052,+0.072, paired over 9 seeds] — headroom unlocked purely by solution diversity / coverage
- **majority@N − pass@1 = -0.017** [95% CI -0.023,-0.010] — net effect of strict voting; CI spanning/below 0 ⇒ majority voting does NOT beat a single agent (it can outvote correct minorities). The coverage gain needs a better selector than vote.

## B. Solution diversity (pairwise token-Jaccard distance)

- all agent pairs, all tasks: mean=0.127, median=0.096
- among pairs where BOTH agents were correct: mean=0.113 (n=6444) — diversity persists even on the same solved problem ⇒ agents reach correctness by different code, not one canonical answer.

## C. Can agent identity be recovered from solution style?

- RF 5-fold accuracy: 0.246 | majority-class chance: 0.250 | permutation-null mean: 0.249 (95th=0.260)
- permutation p (acc ≥ observed under shuffled labels): 0.540
- **Verdict:** indistinguishable from chance ⇒ **no stylistic agent signature**: the null extends beyond success rate to coding style.
