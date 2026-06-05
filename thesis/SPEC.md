# SPEC: Dissertação - Limiar de Capacidade para Emergência em SLMs

**Data**: 2026-04-25
**Status**: Em progresso
**Formato**: Markdown → ABNT UERJ (conversão futura)
**Idioma**: Português

---

## 1. Reframing da Pesquisa

### 1.1 Problema Original (Jan 2026)
> "Podem SLMs idênticos desenvolver especialização emergente através de acumulação de contexto?"

### 1.2 Problema Reframado (Abr 2026)
> "Existe um limiar de capacidade para emergência de especialização, e feedback verificável pode quebrá-lo?"

### 1.3 Motivação para o Reframing
- Riedl (arXiv:2510.05174, ICLR 2026): Documenta falha de emergência em Llama-8B
- Dochkina (arXiv:2603.28990, Mar 2026): Auto-organização prejudica desempenho abaixo de ~10B
- Ambos usaram feedback sintético/LLM-judge; nenhum usou testes unitários

### 1.4 Research Questions (Revisadas)

| RQ | Questão | Respondível com dados atuais? |
|----|---------|------------------------------|
| RQ1 | Existe um limiar de capacidade abaixo do qual a especialização emergente falha em populações de SLMs idênticos? | Sim (1.5B vs 7B) |
| RQ2 | Quais mecanismos (temperatura do roteador, tamanho da população) modulam a emergência de especialização? | Sim (low temp, 5 agents) |
| RQ3 | O feedback verificável (testes unitários) possibilita emergência onde tarefas sintéticas falham? | Parcial (HumanEval tem testes) |

---

## 2. Estrutura da Dissertação

```
1. INTRODUÇÃO
   - Contexto e motivação
   - Problema de pesquisa (capability wall)
   - Objetivos e RQs
   - Contribuições
   - Organização do trabalho

2. REVISÃO DA LITERATURA ✅ REDIGIDO
   2.1 Modelos de Linguagem Pequenos (SLMs)
   2.2 Sistemas Multi-Agente com LLMs
   2.3 Emergência e Auto-Organização
   2.4 O Limiar de Capacidade (Riedl, Dochkina)
   2.5 Benchmarks de Código
   2.6 Síntese e Lacunas

3. METODOLOGIA ✅ REDIGIDO
   3.1 Visão Geral
   3.2 Arquitetura do Sistema
   3.3 Métricas de Emergência (S, D, F)
   3.4 Design Experimental
   3.5 Configuração de Hardware e Software

4. RESULTADOS ✅ REDIGIDO (pendente: 3B, mixed-effects)
   4.1 Visão Geral dos Experimentos
   4.2 O Limiar de Capacidade
   4.3 Mecanismos Moduladores da Emergência
   4.4 Análise Estatística
   4.5 Síntese dos Resultados

5. DISCUSSÃO ⏳ PENDENTE
   5.1 Interpretação dos Resultados
   5.2 Relação com Literatura (Riedl, Dochkina)
   5.3 Limitações
   5.4 Implicações Práticas

6. CONCLUSÃO ⏳ PENDENTE
   - Síntese das contribuições
   - Trabalhos futuros
```

---

## 3. Métricas - Reframing Crítico

### 3.1 Problema
As métricas S, D, F não são novas — são adaptações de métricas estabelecidas:

| Métrica | Equivalente na Literatura |
|---------|--------------------------|
| S (Specialization Index) | Coeficiente de Theil (1970), H₂' de Blüthgen (2006), ROMA (Wang 2020) |
| D (Context Divergence) | Vendi Score (Friedman & Dieng 2022), SEI (arXiv:2510.07888) |
| F (Functional Differentiation) | Chi-square padrão; Riedl usa mixed-effects + PID |

### 3.2 Solução
**NÃO** reivindicar novidade. Framing correto:
> "Adaptamos métricas clássicas de teoria da informação e ecologia quantitativa para o contexto de agentes LLM com acumulação de contexto."

### 3.3 Upgrade: Mixed-Effects Model
- **Status**: A implementar (~2-4 horas)
- **Justificativa**: Alinha com Riedl (2025), mais robusto que chi-square
- **Modelo**: `sucesso ~ tipo_tarefa * agente + (1|problema)`

---

## 4. Experimentos

### 4.1 Completos

| Experimento | Modelo | Agentes | Temp | Pass@1 | S | Significativo |
|-------------|--------|---------|------|--------|------|---------------|
| exp2.1_experimental | 1.5B | 3 | 0.5 | 57.3% | 0.009 | Não |
| exp_low_temp | 1.5B | 3 | 0.1 | 53.7% | 0.196 | Sim |
| exp_5_agents | 1.5B | 5 | 0.3 | 62.2% | 0.133 | Sim |
| exp_7b_model | 7B | 3 | 0.3 | 84.8% | 0.116 | Sim |

### 4.2 Pendentes

| Experimento | Status | Ação |
|-------------|--------|------|
| exp_3b_model | Config criado | Download modelo + executar |
| Mixed-effects analysis | Não implementado | Implementar + re-rodar em todos |

### 4.3 Comandos

**Download modelo 3B:**
```bash
huggingface-cli download Qwen/Qwen2.5-Coder-3B-Instruct-GGUF \
  qwen2.5-coder-3b-instruct-q4_k_m.gguf \
  --local-dir models/
```

**Executar experimento 3B:**
```bash
python scripts/run_experiment.py --config config/exp_3b_model.yaml --seeds 42
```

---

## 5. Citações Críticas a Adicionar

### 5.1 Tier 1 (Essenciais)
- [ ] Riedl (arXiv:2510.05174, ICLR 2026) - Framework de emergência
- [ ] Dochkina (arXiv:2603.28990, Mar 2026) - Limiar de capacidade
- [ ] La Malfa et al. (arXiv:2505.21298) - "LLMs Miss the Multi-Agent Mark"
- [ ] Rahman & Schranz (arXiv:2506.14496) - Swarms: frontier or stretch?
- [ ] Iwasaki et al. (arXiv:2509.04537) - El Farol Bar, diferenciação por memória

### 5.2 Tier 2 (Mecanismos)
- [ ] Theil (1970) - Coeficiente de incerteza original
- [ ] Blüthgen et al. (2006) - H₂' em ecologia
- [ ] Wang et al. (ICML 2020) - ROMA, diversidade em MARL
- [ ] Friedman & Dieng (2022) - Vendi Score

### 5.3 Tier 3 (Contexto)
- [ ] Choi et al. (NeurIPS 2025) - "Debate or Vote"
- [ ] Zhang et al. (arXiv:2502.08788) - "Stop Overvaluing Multi-Agent Debate"
- [ ] Belcak & Heinrich (2025) - NVIDIA SLM position paper

---

## 6. Arquivos Criados

```
thesis/
├── SPEC.md                 ← Este arquivo
├── cap2_revisao.md         ← Capítulo 2: Revisão da Literatura
├── cap3_metodologia.md     ← Capítulo 3: Metodologia
└── cap4_resultados.md      ← Capítulo 4: Resultados

config/
└── exp_3b_model.yaml       ← Config para experimento 3B
```

---

## 7. Checklist de Ações

### 7.1 Imediato (Esta Semana)

- [x] Download Qwen2.5-Coder-3B-Instruct-GGUF ← **EM PROGRESSO (background)**
- [ ] Executar exp_3b_model.yaml ← **AGUARDANDO DOWNLOAD**
- [ ] Atualizar cap4_resultados.md com dados do 3B
- [ ] Implementar mixed-effects model (~2-4 horas)
- [ ] Re-rodar análise de F em todos os experimentos

**Comando para executar após download:**
```bash
python scripts/run_experiment.py --config config/exp_3b_model.yaml --seeds 42
```

### 7.2 Curto Prazo (Próxima Semana)

- [ ] Redigir Capítulo 5 (Discussão)
- [ ] Redigir Capítulo 6 (Conclusão)
- [ ] Redigir Capítulo 1 (Introdução)
- [ ] Compilar bibliografia com citações Tier 1-3

### 7.3 Médio Prazo (Antes da Defesa)

- [ ] Converter Markdown → LaTeX/Word ABNT UERJ
- [ ] Gerar figuras (gráficos de S over time, heatmaps de afinidade)
- [ ] Revisão com orientador
- [ ] Ajustes finais

---

## 11. Notas Pendentes

### 11.1 Mixed-Effects Implementation

**Arquivo a criar:** `scripts/analyze_mixed_effects.py`

**Dependências:**
```bash
pip install statsmodels
```

**Código base:**
```python
import pandas as pd
import statsmodels.formula.api as smf
import json
from pathlib import Path

def run_mixed_effects(results_dir: str) -> dict:
    """
    Run mixed-effects model on experiment results.
    Model: success ~ task_type * agent + (1|problem)
    """
    # Load task log from results
    results_path = Path(results_dir) / "seed_42" / "task_log.json"
    with open(results_path) as f:
        task_log = json.load(f)

    df = pd.DataFrame(task_log)

    # Fit mixed-effects model
    model = smf.mixedlm(
        "success ~ C(task_type) * C(agent_id)",
        data=df,
        groups=df["problem_id"]
    )
    result = model.fit()

    # Extract key statistics
    return {
        "converged": result.converged,
        "llf": result.llf,
        "aic": result.aic,
        "bic": result.bic,
        "random_effects_var": result.cov_re.iloc[0, 0],
        "icc": result.cov_re.iloc[0, 0] / (result.cov_re.iloc[0, 0] + result.scale),
        "interaction_pvalue": None,  # Extract from result.pvalues
        "summary": str(result.summary())
    }

if __name__ == "__main__":
    experiments = [
        "results/exp2.1_experimental",
        "results/exp_low_temp",
        "results/exp_5_agents",
        "results/exp_7b_model",
        "results/exp_3b_model",
    ]

    for exp in experiments:
        print(f"\n{'='*60}")
        print(f"Experiment: {exp}")
        print('='*60)
        try:
            results = run_mixed_effects(exp)
            print(f"ICC: {results['icc']:.4f}")
            print(f"Converged: {results['converged']}")
        except Exception as e:
            print(f"Error: {e}")
```

**Nota:** O task_log.json pode não existir no formato esperado. Verificar estrutura dos resultados antes de implementar.

### 11.2 Capítulos Pendentes

**Capítulo 1 (Introdução):** ~1500 palavras
- Contextualização do problema
- Justificativa (acessibilidade, custos de API)
- Objetivos geral e específicos
- Contribuições esperadas
- Organização do documento

**Capítulo 5 (Discussão):** ~2000 palavras
- Interpretação dos resultados à luz de Riedl/Dochkina
- O que o limiar significa na prática
- Por que 1.5B falha e 7B funciona (hipóteses)
- Limitações metodológicas
- Implicações para sistemas acessíveis

**Capítulo 6 (Conclusão):** ~800 palavras
- Síntese das contribuições
- Respostas às RQs
- Trabalhos futuros (SWE-bench, mais modelos, análise temporal)

### 11.3 Bibliografia - Entradas a Adicionar

```bibtex
@inproceedings{riedl2026emergent,
  title={Emergent Coordination in Multi-Agent Language Models},
  author={Riedl, C. and others},
  booktitle={ICLR},
  year={2026},
  note={arXiv:2510.05174}
}

@article{dochkina2026capability,
  title={The Capability Threshold for Self-Organization in LLM Populations},
  author={Dochkina, A. and others},
  journal={arXiv preprint arXiv:2603.28990},
  year={2026}
}

@article{lamalfa2025miss,
  title={Large Language Models Miss the Multi-Agent Mark},
  author={La Malfa, E. and others},
  journal={arXiv preprint arXiv:2505.21298},
  year={2025}
}

@article{theil1970uncertainty,
  title={On the estimation of relationships involving qualitative variables},
  author={Theil, Henri},
  journal={American Journal of Sociology},
  volume={76},
  number={1},
  pages={103--154},
  year={1970}
}

@article{bluthgen2006measuring,
  title={Measuring specialization in species interaction networks},
  author={Bl{\"u}thgen, Nico and others},
  journal={BMC Ecology},
  volume={6},
  number={1},
  pages={1--12},
  year={2006}
}
```

### 11.4 Figuras a Gerar

1. **Arquitetura do Sistema** (já em ASCII no cap3)
   - Converter para figura vetorial se necessário

2. **Heatmap de Afinidade** por experimento
   - Eixo X: tipos de tarefa
   - Eixo Y: agentes
   - Cor: taxa de sucesso

3. **Gráfico de Barras: S por Experimento**
   - Com barras de erro (bootstrap)
   - Linha horizontal indicando threshold de significância

4. **Scatter: S vs Pass@1**
   - Um ponto por experimento
   - Identificar trade-offs

---

## 8. Decisões Tomadas

| Decisão | Escolha | Justificativa |
|---------|---------|---------------|
| Benchmark | HumanEval (não SWE-bench) | Variância preservada para SLMs, tempo de execução |
| Modelos | Qwen2.5-Coder 1.5B/3B/7B | Licença, família completa, GGUF disponível |
| Métricas | S, D, F (creditadas) | Estabelecidas, interpretáveis, comparáveis |
| F upgrade | Mixed-effects | Alinha com Riedl, mais robusto |
| RQs | 3 focadas no limiar | Respondíveis com dados existentes |
| Framing | Limiar de capacidade | Posiciona vs. Riedl/Dochkina, permite null result |

---

## 9. Riscos e Mitigações

| Risco | Probabilidade | Mitigação |
|-------|---------------|-----------|
| 3B não mostra emergência | Média | Resultado ainda é publicável (confirma limiar entre 3B-7B) |
| Mixed-effects não converge | Baixa | Fallback para chi-square (já temos) |
| Tempo insuficiente para SWE-bench | Alta | Declarar como trabalho futuro |
| Banca questiona métricas | Média | Créditos claros + limitações documentadas |

---

## 10. Próxima Sessão

Ao retomar, executar na ordem:

1. `huggingface-cli download ...` (modelo 3B)
2. `python scripts/run_experiment.py --config config/exp_3b_model.yaml --seeds 42`
3. Implementar mixed-effects em `scripts/analyze_mixed_effects.py`
4. Atualizar `thesis/cap4_resultados.md` com novos dados
5. Continuar para Capítulos 5 e 6

---

*Última atualização: 2026-04-25*
