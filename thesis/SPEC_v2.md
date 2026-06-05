# SPEC v2: Dissertação - Limiar de Capacidade para Emergência em SLMs

**Data**: 2026-04-25 (atualizado)
**Status**: Em progresso
**Baseado em**: Análise atualizada do capability wall (Abr 2026)

---

## 0. Correções Críticas da Literatura

### 0.1 Erros de Citação a Corrigir

| Citação Original | Correção |
|------------------|----------|
| Choi et al. "Debate or Vote" (OpenReview iUjGNJzrF1) | **arXiv:2508.17536** |
| Iwasaki et al. El Farol Bar (arXiv:2509.04537) | **Takata, Masumori, e Ikegami** |
| Dochkina (arXiv:2603.28990) como resultado estabelecido | **Preprint MIPT, autor único, citar como "claims to be tested"** |

### 0.2 Problemas com Dochkina (2603.28990)
- Preprint sem peer review
- Referencia modelos futuros ("GPT-5.4", "Gemini-3-flash") que não existem em abril 2026
- Effect sizes específicos (Cohen's d=1.86, +14%, +44%) devem ser tratados como hipóteses, não resultados

---

## 1. O Capability Wall Revisado

### 1.1 Não é UM Wall — São TRÊS

O limiar de capacidade é uma **pilha de três thresholds compostos**:

| Nível | Mecanismo de Falha | Evidência |
|-------|-------------------|-----------|
| **Agent-level** | Fragilidade de formato/protocolo | MapCoder-Lite: 7B colapsa sem distilação (13.2% → 28.3%) |
| **Coordination-level** | Colapso de Theory-of-Mind | Riedl: Llama-8B 10-14% sucesso, sinergia espúria temporal |
| **System-level** | Overhead > Ganhos | Rahman & Schranz: 300× mais tempo que Boids clássico |

### 1.2 Localização do Wall

**Convergência empírica: 7B-8B para tarefas de coordenação**

- Abaixo: Multi-agent frequentemente *pior* que single-agent
- Acima (com protocolo adequado): Multi-agent começa a agregar valor

### 1.3 O Wall é Permeável

Evidências de que SLMs podem cruzar o wall com intervenções específicas:

| Trabalho | Modelo | Resultado | Mecanismo |
|----------|--------|-----------|-----------|
| MALT (arXiv:2412.01928) | 1.5B Qwen | Bate Llama-70B em GSM-Symbolic | Role-specific preference pairs |
| AgentPRM (arXiv:2502.10325) | 3B | Bate GPT-4o em ALFWorld | Process rewards |
| Berkeley TinyAgent | 1.1B | 78.9% function-calling | Fine-tuning direcionado |

**Implicação para a tese:** O wall é real para SLMs off-the-shelf em configurações sem tuning. É permeável com post-training direcionado.

---

## 2. Benchmark Suite Atualizada

### 2.1 O Que Deve Ser ABANDONADO

| Benchmark | Motivo |
|-----------|--------|
| HumanEval vanilla | Saturado, contaminado (~25% overlap GPT-4) |
| MBPP vanilla | Saturado, contaminado |
| SWE-bench Lite | **Oficialmente superseded** por SWE-bench Verified |
| APPS, CodeContests | Tratados como training data por modelos modernos |

### 2.2 Suite Recomendada (Abril 2026)

**Tier 1 — Primário, contamination-aware:**

| Benchmark | Tamanho | SLM Range | GPU-hours (7B) |
|-----------|---------|-----------|----------------|
| LiveCodeBench v6 | 1000+ | 15-40% | 3-5h |
| BigCodeBench-Hard | 148 | 25-40% | 1-2h |
| SWE-bench Verified | 500 | <5% | 30-100h |

**Tier 2 — Diagnóstico:**

| Benchmark | Propósito |
|-----------|-----------|
| HumanEval+ / MBPP+ (EvalPlus) | Smoke test, paridade com literatura |
| CRUXEval-O | Code reasoning |
| Terminal-Bench 2.0 | Se tiver terminal harness |

**Tier 3 — Multi-agent específico:**

| Benchmark | Nota |
|-----------|------|
| MultiAgentBench / MARBLE (ACL 2025) | **Único benchmark com KPIs por-agente e comparação de topologias** |

### 2.3 Decisão para a Dissertação

**Dado o constraint de calendário:**

- ✅ **Manter**: HumanEval (dados já coletados, variância preservada para SLMs)
- ✅ **Adicionar**: Framing como "EvalPlus smoke test"
- ⚠️ **Reconhecer**: Limitações de contaminação na seção de limitações
- 📝 **Futuro**: LiveCodeBench, SWE-bench Verified como trabalho futuro

---

## 3. Metodologia PID - Upgrade

### 3.1 Por Que PID é a Escolha Certa

- Sobrevive à crítica de Schaeffer (métrica contínua, não accuracy-based)
- Fornece testes de significância falsificáveis
- Separa sinergia de redundância

### 3.2 Implementação Riedl

| Componente | Detalhe |
|------------|---------|
| PID variant | Williams-Beer I_min (+ MMI como robustez) |
| Discretização | Quantile-binned agent deviations |
| Correção de viés | Miller-Madow |
| Nulls | Row-shuffle + Column-shuffle, combinados via Fisher's method |
| Teste de coalizão | G3 (whole-minus-best-pair) |

### 3.3 Decisão para a Dissertação

**Dado o constraint de calendário:**

- ✅ **Manter**: S (Theil/Blüthgen) - já implementado, funciona
- ✅ **Adicionar**: Mixed-effects para F
- ⚠️ **Reconhecer**: PID é mais principiado, indicar como trabalho futuro
- 📝 **Citar**: Riedl como "state of the art metodológico"

---

## 4. Contradições a Engajar (Não Esconder)

### 4.1 Belcak vs. Literatura de Falhas

| Posição | Evidência |
|---------|-----------|
| **Belcak (NVIDIA):** "SLMs são suficientes para tarefas agênticas" | 10-30× redução de custo |
| **Contra:** MapCoder-Lite, Rahman & Schranz, Cemri | 41-86.7% failure rates, 300× overhead |

**Reconciliação:** Belcak vale para *single-agent function calling* com schema constraints. Não vale para *multi-agent coordination* com contexto compartilhado.

### 4.2 RLVR vs. Judge Rewards (Inversão Surpreendente)

| Escala | Melhor Reward |
|--------|---------------|
| 125M-350M | **Judge rewards** (+5-10 pontos vs verifiable) |
| 6.7B | Empate (~1.5 pontos diferença) |

**Implicação:** Modelos pequenos precisam de *signal density*, não *signal correctness*. Verifiable rewards sozinhos são insuficientes.

### 4.3 AgentGroupChat-V2 vs. Debate-as-Martingale

| Trabalho | Claim |
|----------|-------|
| AgentGroupChat-V2 | +11pp em MATH-L5 com multi-agent |
| Choi (NeurIPS 2025) | Debate é martingale, não melhora expected correctness |

**Reconciliação:** AgentGroupChat usa task decomposition + heterogeneous backbones, não debate puro. Ganhos vêm de majority voting + heterogeneidade.

---

## 5. Posição Defensável da Tese

### 5.1 Cinco Claims Suportados pela Literatura

1. **O wall é task-conditional**: Agent-level (ToM), coordination-level (format), system-level (overhead) compõem abaixo de 7B

2. **PID é o framework de medição correto**: Sobrevive crítica de Schaeffer, fornece null-model significance tests

3. **Verifiable rewards sozinhos são insuficientes**: Credit-assignment (M-GRPO, CCPO) e process rewards (AgentPRM) importam mais que verifiability

4. **O wall está descendo ~10× a cada 3 anos**: Qualquer threshold específico deve ser hedged contra este drift

5. **Não existem benchmarks multi-agent SLM específicos**: Esta é uma lacuna que a tese pode documentar

### 5.2 Contribuição Plausível

> "Estudo empírico cuidadoso de como especialização de papéis emerge em swarms SLM, medido com ferramentas information-theoretic, validado contra benchmark contamination-aware."

---

## 6. Arquivos Atualizados

### 6.1 Estrutura Atual

```
thesis/
├── SPEC.md                 ← v1 (manter como histórico)
├── SPEC_v2.md              ← ESTE ARQUIVO
├── cap2_revisao.md         ← PRECISA ATUALIZAÇÃO (citações)
├── cap3_metodologia.md     ← OK (já credita métricas)
└── cap4_resultados.md      ← PRECISA ATUALIZAÇÃO (3B pending)

config/
└── exp_3b_model.yaml       ← EXECUTANDO
```

### 6.2 Atualizações Necessárias em cap2_revisao.md

1. Corrigir citação Choi → arXiv:2508.17536
2. Corrigir citação El Farol → Takata et al.
3. Adicionar caveat sobre Dochkina (preprint, citar com cautela)
4. Adicionar seção sobre contradições na literatura
5. Expandir seção 2.4 com os três níveis do wall

---

## 7. Checklist Atualizada

### 7.1 Imediato

- [x] Download Qwen2.5-Coder-3B
- [x] Iniciar exp_3b_model.yaml (background task be22616)
- [ ] Atualizar cap2 com correções de citação
- [ ] Implementar mixed-effects model

### 7.2 Curto Prazo

- [ ] Atualizar cap4 com resultados 3B
- [ ] Adicionar seção de contradições na literatura
- [ ] Redigir Cap 5 (Discussão) com engajamento das contradições
- [ ] Redigir Cap 6 (Conclusão)
- [ ] Redigir Cap 1 (Introdução)

### 7.3 Limitações a Documentar Explicitamente

- [ ] Benchmark saturation/contamination do HumanEval
- [ ] PID não implementado (Riedl é state-of-art, nós usamos versão simplificada)
- [ ] Single-seed experiments (futuros devem usar multi-seed)
- [ ] Threshold drift (~10× a cada 3 anos)

---

## 8. Citações Corrigidas

### 8.1 Tier 1 (Essenciais) - CORRIGIDO

```bibtex
@inproceedings{riedl2026emergent,
  title={Emergent Coordination in Multi-Agent Language Models},
  author={Riedl, C.},
  booktitle={ICLR},
  year={2026},
  note={arXiv:2510.05174. Single-author, Llama-8B results.}
}

@article{dochkina2026capability,
  title={The Capability Threshold for Self-Organization},
  author={Dochkina, A.},
  journal={arXiv preprint arXiv:2603.28990},
  year={2026},
  note={CAUTION: Single-author MIPT preprint, references future models.
        Cite as claims to be tested, not established results.}
}

@inproceedings{choi2025debate,
  title={Debate or Vote? Deliberation vs. Aggregation in LLMs},
  author={Choi, M. and others},
  booktitle={NeurIPS},
  year={2025},
  note={arXiv:2508.17536. Spotlight. Proves debate is martingale.}
}

@inproceedings{takata2025elfarol,
  title={Emergent Individuality in LLM Populations: El Farol Bar Problem},
  author={Takata, S. and Masumori, A. and Ikegami, T.},
  booktitle={ALIFE},
  year={2025},
  note={arXiv:2509.04537. NOT Iwasaki et al.}
}
```

### 8.2 Tier 1 - Adições Críticas

```bibtex
@inproceedings{cemri2025mast,
  title={MAST: A Multi-Agent System Taxonomy for LLM Failures},
  author={Cemri, E. and others},
  booktitle={NeurIPS D\&B},
  year={2025},
  note={arXiv:2503.13657. 14 failure modes, 41-86.7\% failure rates.}
}

@article{mapcoderlite2025,
  title={MapCoder-Lite: Lightweight Multi-Agent Code Generation},
  author={Various},
  journal={arXiv preprint arXiv:2509.17489},
  year={2025},
  note={Shows 7B collapse without distillation: 13.2\% → 28.3\%.}
}

@article{rahman2025swarms,
  title={LLM-Powered Swarms: A New Frontier or a Conceptual Stretch?},
  author={Rahman, A. and Schranz, M.},
  journal={arXiv preprint arXiv:2506.14496},
  year={2025},
  note={300× overhead for LLM-Boids vs classical.}
}
```

---

## 9. Experimento 3B - Status

**Task ID:** be22616
**Status:** Running
**Output:** /tmp/claude-1000/-home-vlofgren-Projects-mestrado-dissertacao/tasks/be22616.output

**Comando para verificar:**
```bash
tail -f /tmp/claude-1000/-home-vlofgren-Projects-mestrado-dissertacao/tasks/be22616.output
```

**Após conclusão:**
1. Ler results/exp_3b_model/seed_42/final_metrics.json
2. Atualizar cap4_resultados.md seção 4.2.3
3. Atualizar tabela resumo na seção 4.1

---

*SPEC v2 - Atualizado 2026-04-25 baseado em análise de capability wall*
