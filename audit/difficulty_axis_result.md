# Label-free difficulty-axis interaction test — RESULT (2026-06-09)

> **VERDICT:** The functional-differentiation null is **NOT an artifact of the 4-way task
> taxonomy.** On a fully label-free continuous difficulty axis (AST size / cyclomatic / length,
> the features pre-registered for the MBPP axis), the **agent×difficulty interaction is null in
> every ensemble run** under task-clustered inference. Delivers the robustness test cap5 §5.5.2
> pre-registered (`data/mbpp_difficulty_axis.json` was built but never run). Closes the construct-
> validity gap: the type-based null does not hinge on the noisy `logic` fallback cell.

Script: `audit/difficulty_axis_interaction.py` (0-GPU, consumes existing ensemble logs).
Axis: `data/humaneval_difficulty_axis.json` (164 problems; difficulty_z = mean of z-scored
prompt_chars, prompt_tokens, ast_nodes, cyclomatic, canon_lines; terciles 55/54/55).

## Result — GEE logit `success ~ C(agent_id) * difficulty_z`, cov=exchangeable, groups=task_id

| Condition | agents | n_obs | agent MAIN-effect p | agent×difficulty INTERACTION p | verdict |
|---|---|---|---|---|---|
| het_swarm_ensemble [2×1.5B+1×3B] | 3 | 4920 | **0.0013** | **0.631** | competence main effect only; NO difficulty interaction |
| ens_3b_n3 [3×3B] | 3 | 4920 | 0.761 | 0.228 | null/null |
| ens_3b [4×3B] | 4 | 6560 | 0.468 | 0.091 | null (interaction closest but NS) |
| ens_7b [4×7B] | 4 | 3280 | 0.975 | 0.955 | null/null |

## Interpretation
- **Identical-agent ensembles (ens_3b, ens_7b, ens_3b_n3): no agent×difficulty interaction** (all
  p ≥ 0.09) — mirrors the agent×type null. The functional-differentiation null is robust to dropping
  the hand taxonomy entirely; it is not a `logic`-fallback artifact.
- **Heterogeneous swarm: a competence MAIN effect (p=0.0013, the 3B is uniformly better) with NO
  difficulty interaction (p=0.63).** Identical structure to the agent×type finding: scale buys
  uniform competence, not a difficulty- (or type-) dependent role. A bigger model is not
  *relatively* better on hard problems here — it is better everywhere by a constant margin.
- This is the label-free analogue of `het_interaction_result.md` v3 and corroborates it.

## Note on the MBPP axis
`data/mbpp_difficulty_axis.json` has a tercile bug (`medium`=0; only easy/hard populated). The
HumanEval axis built here computes terciles correctly (qcut → 55/54/55). The MBPP axis should be
regenerated with the same correct tercile logic if it is ever used (primary analyses use the
continuous `difficulty_z`, so this does not affect the result above).

## Thesis impact
cap5 §5.5.2 can change "pré-registramos … como teste primário" (planned) → "executamos … o teste
de eixo latente livre de rótulos confirma o nulo (interação agente×dificuldade NS em todos os
ensembles; ver audit/difficulty_axis_result.md)" — i.e. a DELIVERED robustness result, not a promise.
