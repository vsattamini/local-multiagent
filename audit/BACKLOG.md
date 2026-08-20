# Deferred review streams — CHECK LATER (registered 2026-06-09)

Two review threads were started, then deliberately PARKED by the user to revisit later. Neither has
been acted on. Partial agent findings are captured below so each can be resumed without re-running.
Nothing here has been applied to the thesis or code.

---

## POINT 1 — Metrics & harness critique (fan-out suspended mid-run)

Two agents (metrics-with-websearch; harness/engineering) were launched and stopped before finishing.
Revisit when ready to harden metrics/infrastructure.

**Metrics critique — to resume:** systematically interrogate S / D / F / any@N / load-Gini vs
established literature (ecology, economics, MARL, diversity).
- Partial signal before stop: the agent was converging on **Blüthgen's H2′ / d′ (BMC Ecology 2006,
  already cited in cap3) as a possibly better-justified, bias-corrected specialization index than
  Theil's uncertainty coefficient U** (current S). Worth checking whether reporting H2′/d′ alongside S
  changes or merely hardens the allocation story. Other candidates to compare: Gini/HHI, normalized MI,
  Pielou evenness (for S); Vendi score / DPP (for D); PID-synergy (Riedl) and crossed-RE GLMM (for F).

**Harness critique — to resume:** forward-looking design improvements (correctness already audited in
`audit/CODE_AUDIT.md` — do NOT re-audit bugs). Axes flagged before stop:
- the affinity router's 0.5 prior for untried types may make early routing ~random → "allocation"
  partly a router tautology;
- the harness FORCES assignment (no abstention/refusal) → structurally blind to the niche-via-
  abstention that Dochkina measures;
- single-assignment (LR non-computable) vs ensemble (S≡0) never co-observe allocation+differentiation;
- **the harness should DETECT CUDA-init failure and ABORT rather than silently fall back to CPU**
  (the 2026-06-09 suspend dropped het_15_7b to CPU at 6.7 tok/s — a real incident this session);
- log more per step (agent states for PID, abstention events, difficulty) to enable richer analyses.

To relaunch: re-spawn the two agents with the same prompts (metrics-with-websearch; harness-design).

---

## POINT 2 — Research-questions sense-check (do the RQs still hold given the null?)

Three agents launched; two completed before stop (RQ sense/reframe ✅, RQ↔contribution alignment ✅),
one killed (defensibility framing — partial). Revisit when deciding final cap1 framing.

**Key finding (RQ sense-check & honest-reframe agent, completed):** the thesis is *already* integrity-
clean where it matters — cap1 §1.5 keeps H1–H4 verbatim with refuted verdicts; §1.6 contributions are
already reframed around the surviving findings; cap4/5/6 answer the RQs honestly as nulls. **The ONE
remaining seam:** the *Objetivo Geral* (§1.3.1) and RQ2's "**deste limiar**" subclause still
grammatically PRESUPPOSE the refuted threshold ("investigar a localização do limiar" = stating as a goal
the locating of a thing later concluded not to exist).

Recommended minimal edits (DEFERRED — not applied):
- **§1.3.1 Objetivo Geral** → recast from "investigar a localização do limiar" to "investigar **se**
  … desenvolvem especialização emergente; em caso negativo, caracterizar o que de fato emerge
  (alocação) e o que produz variância entre agentes (heterogeneidade de substrato)."
- **§1.3.2 OE-1** "Caracterizar … o limiar" → "**Testar a existência** de um limiar."
- **§1.4 RQ2** delete "deste limiar" → "modulam a auto-organização observada … alocação ou
  diferenciação funcional?"
- **§1.4 RQ3** "abaixo do limiar" → "para induzir especialização funcional em agentes idênticos."
- **RQ1 and H1–H4: keep VERBATIM** — keeping a question whose answer is "no" *is* the honest move;
  retiring RQ1 would be the dishonest one.
- Optional **RQ4 (NEW), tagged "(questão emergente da refutação de RQ1)"** — "qual é a fonte da
  variância entre agentes — emergência ou substrato?" The retrospective tag is non-negotiable to avoid
  HARKing. HARK exposure is zero if (i) H1–H4 stay frozen, (ii) RQ4 is tagged retrospective, (iii) the
  title headlines the *allocation-vs-differentiation distinction* (a method contribution, conventionally
  retrospective), not a threshold or emergent roles.
- **Title direction:** "Alocação, não especialização: um nulo bem-potente sobre diferenciação funcional
  emergente em enxames de SLMs idênticos" (or a contrastive "limiar testado-e-refutado" framing).

**Key finding (RQ→contribution alignment agent, completed):** the RQ scaffolding cleanly governs the
*empirical null* (RQ1/RQ3 → H1/H4 → §4.2/§4.4 → §6.1 → contribution #1; no verdict drift), BUT **two of
the four prized contributions are ORPHANS — governed by no RQ:**
- **Contribution #2 (allocation-vs-differentiation distinction + cluster-robust/task_id inference recipe)**
  — the thesis's self-declared *most durable* contribution (cap6 §6.4), yet no RQ asks about unit of
  inference / distinguishing routing-concentration from specialization. Fix = **add RQ4** on measurement
  & inference unit (latent already in §1.3.2 obj. 3 "validar métricas / framework reproduzível").
- **Contribution #4 (heterogeneity = uniform competence + +4.9pp ensemble coverage)** — RQ1–3 are scoped
  to *identical* SLMs; the het test arises in §4.5 as "a pergunta natural", ungoverned. Fix = **extend
  RQ2 or add RQ5**: "se a diferenciação não emerge entre idênticos, de onde vem o valor multi-agente?"
- **Difficulty-axis result is delivered but UNDER-CREDITED** — not named in §1.6/§6.2; and cap5 §5.5.2
  still says "pré-registramos … teste primário" (planned) though it was EXECUTED (see
  `audit/difficulty_axis_result.md`). Fix = credit it under contribution #1 + change §5.5.2 planned→executed.
- Coverage verdict: the RQs currently circumscribe only ~⅓–½ of what the thesis delivers; center of
  gravity moved to method + heterogeneity + coverage.

**Combined deferred cap1 edit plan (sense-check + alignment agents agree):** keep RQ1/RQ3 + H1–H4 verbatim;
de-presuppose Objetivo Geral §1.3.1, OE-1 §1.3.2, RQ2 ("deste limiar"), RQ3 ("abaixo do limiar"); add
**RQ4 (method/inference, tagged retrospective)** and **RQ5/extended-RQ2 (source-of-value/heterogeneity)**;
credit the difficulty-axis result + fix §5.5.2 tense; retitle around allocation-vs-differentiation. HARK
risk zero if H1–H4 frozen and new RQs tagged as emergent-from-refutation.

**Still not done (one agent killed before finishing):** the null-result DEFENSE MEMO — elevator defense,
keep-vs-reframe recommendation, talking points for the two soft spots (disjoint designs / xval_roles
unrun; the Dochkina arXiv:2603.28990 contradiction). To resume: re-spawn the defensibility agent.
