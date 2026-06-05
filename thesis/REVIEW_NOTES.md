# Notas Mestres para Revisão da Dissertação

Este documento consolida as principais decisões pendentes, verificações necessárias e tensões interpretativas identificadas durante a redação. Use como checklist para a passada de revisão.

---

## 1. Decisões Narrativas Pendentes

### 1.1 Tom em relação a Dochkina (2026)
**Onde aparece:** cap1 §1.2, cap2 §2.4.2, cap5 §5.2.2, cap6 §6.1.1

**Tensão:** Preprint MIPT, autor único, referencia modelos não-existentes. Mas a hipótese qualitativa é consistente com nossos dados.

**Opções:**
- (a) Manter citação com qualificação ("hipótese ainda não revisada por pares") — versão atual
- (b) Remover toda menção até peer review
- (c) Citar apenas como "literatura cinza" em nota de rodapé

**Recomendação atual:** (a) — equilibra cautela com aproveitamento da motivação teórica.

---

### 1.2 Como apresentar a refutação de H4
**Hipótese H4:** "Feedback verificável binário é mais eficaz que feedback escalar para induzir emergência"

**Resultado:** Refutada parcialmente — feedback verificável não foi suficiente para quebrar o limiar.

**Opções:**
- (a) Manter H4 forte e celebrar a refutação como achado positivo
- (b) Reescrever H4 mais cautelosa: "Feedback verificável é necessário, possivelmente suficiente em modelos próximos ao limiar"
- (c) Remover H4 das hipóteses e tratar como "questão aberta" no cap 1

**Recomendação atual:** (a) — refutação de hipótese é resultado científico válido.

---

### 1.3 Concentração de roteamento vs. especialização genuína
**Onde aparece:** cap4 §4.4.2, cap5 §5.1.2

**Tensão:** O teste LR mostra que apenas 7B tem diferenciação funcional genuína. Mas o 3B com baixa temperatura tem S=0.390 (maior que tudo). Como reconciliar?

**Possíveis interpretações:**
- (a) A separação perfeita IMPEDE o teste LR — especialização pode ser real mas indetectável formalmente
- (b) "Especialização" no 3B com baixa temperatura é principalmente concentração de carga (similar ao 1.5B com baixa temp)
- (c) Híbrido: parcialmente concentração + parcialmente diferenciação

**Recomendação:** Adicionar seção 4.4.4 dedicada a esta questão interpretativa.

---

## 2. Verificações Bibliográficas

### 2.1 Citações que precisam de checagem
- [ ] Choi et al. → arXiv:2508.17536 (NeurIPS 2025 Spotlight)
- [ ] Takata, Masumori & Ikegami → arXiv:2509.04537 (NÃO "Iwasaki")
- [ ] Dochkina (2026) → arXiv:2603.28990 (verificar se ainda é preprint)
- [ ] MapCoder-Lite → arXiv:2509.17489
- [ ] Cemri et al. (MAST) → arXiv:2503.13657
- [ ] Riedl (2025) → arXiv:2510.05174 (ICLR 2026)

### 2.2 Citações sem entrada .bib (ainda)
**Falta criar:** `references.bib` com todas as citações. Verificar:
- Brown et al. (2020) — GPT-3
- Bonabeau et al. (1999) — Swarm Intelligence
- Theil (1970) — Uncertainty coefficient
- Blüthgen (2006) — H2' specialization
- Kaplan et al. (2020) — Scaling laws
- Abdin et al. (2024) — Phi-3, Phi-4
- Hong et al. (2024) — MetaGPT
- Wu et al. (2024) — AutoGen
- Qian et al. (2024) — ChatDev
- Li et al. (2023) — CAMEL
- Du et al. (2024) — Multi-agent debate
- Jimenez et al. (2024) — SWE-bench
- Liu et al. (2023) — EvalPlus
- La Malfa et al. (2025) — Multi-agent mark
- Casadei et al. (2023) — Collective intelligence
- Belcak & Heinrich (2025) — NVIDIA SLM paper
- Rahman & Schranz (2025) — LLM-Powered Swarms
- Jimenez-Romero et al. (2025) — emergent swarm
- Chen et al. (2021) — HumanEval
- Austin et al. (2021) — MBPP
- Yang et al. (2023) — contamination

### 2.3 Possíveis citações faltantes
- [ ] Phi-4 (Abdin et al. 2024) — atualização do Phi-3
- [ ] Trabalhos sobre PID (Williams & Beer 2010) se for usar conceito
- [ ] Trabalhos sobre emergence (Wei et al. 2022) — emergent abilities
- [ ] Vendi Score paper (Friedman & Dieng 2023) se mencionar D

---

## 3. Verificações Estatísticas/Técnicas

- [ ] **Múltiplas sementes pendentes**: todos os experimentos rodaram com seed=42 apenas. Replicar com 5-10 sementes para CIs.
- [ ] **Teste LR para casos com separação**: considerar Firth-style penalized regression para conditions com perfeito separation (exp_low_temp, exp_5_agents, exp_3b_low_temp, exp_3b_5_agents)
- [ ] **Verificar fórmulas de S, D, F** em cap3 contra implementação em src/swarm/metrics.py
- [ ] **Sweep results pending**: ~32 experimentos sendo executados — incluir como nova seção 4.6
- [ ] **ICC**: investigar por que estava sempre 0.500 nas primeiras tentativas de GLMM (provavelmente reflexo de design single-obs-per-problem)

---

## 4. Estrutura e Estilo

### 4.1 Sínteses redundantes
Há síntese em **três** lugares:
- cap4 §4.5 (Síntese dos Resultados)
- cap5 §5.6 (Síntese)
- cap6 §6.1 (Resposta às Questões de Pesquisa)

Decidir: eliminar uma, ou diferenciar funções (resultado/discussão/conclusão).

### 4.2 Cap2 §2.6 (Contradições)
Seção experimental, não-tradicional em ABNT. Decidir manter ou diluir.

### 4.3 Cap1 — falta seção?
ABNT UERJ pode exigir "Justificativa" ou "Motivação" como seção dedicada. Verificar manual do programa.

### 4.4 Figuras
Atualmente:
- Cap3 tem figura ASCII da arquitetura → trocar por TikZ/Mermaid
- Cap4 tem várias tabelas → considerar adicionar gráficos (heat maps de afinidade)
- Cap5 — sem figuras

ABNT permite figuras coloridas em versão digital.

### 4.5 Tabelas
- [ ] Numerar tabelas e referenciar no texto consistentemente
- [ ] Adicionar legendas no formato "Tabela N — Descrição" (padrão ABNT)
- [ ] Verificar fonte/origem das tabelas (própria autoria → "Fonte: O autor (2026).")

---

## 5. Pendências Estruturais Maiores

- [ ] **Resumo / Abstract** (PT-BR e EN) — não escrito ainda
- [ ] **Sumário automático** — gerar depois de finalização
- [ ] **Lista de figuras, tabelas, abreviaturas** — gerar
- [ ] **Folha de aprovação, dedicatória, agradecimentos** — preencher
- [ ] **Apêndices** — considerar incluir:
  - A. Configurações YAML completas dos experimentos
  - B. Categorização dos 164 problemas HumanEval em 4 tipos
  - C. Resultados completos dos sweeps (tabelas longas)
  - D. Código-fonte das métricas (S, D, F) com docstrings
- [ ] **Bibliografia** — `references.bib` ainda não criado

---

## 6. Possíveis Pontos de Defesa em Banca

### Argumentos fortes
- Mapeamento empírico em família coerente (Qwen2.5-Coder)
- Distinção LR vs. qui-quadrado (rigor estatístico)
- Achado contraintuitivo (3B baseline também falha)
- Identificação do "ponto sem trade-off" (3B com intervenção)

### Argumentos fracos / pontos de ataque potenciais
- Single-seed (não há replicação)
- Apenas uma família de modelo
- HumanEval pode estar contaminada
- 4 categorias de tarefas é pouco
- Métricas adaptadas — derivação rigorosa não apresentada
- Limiar entre 3B e 7B é quase tautológico (já documentado por Riedl)

### Perguntas prováveis em banca
1. "Por que não usou múltiplas sementes?"
2. "Como sabemos que HumanEval não está nos dados de treino?"
3. "O que justifica focar em Qwen e não em uma família mais conhecida?"
4. "A 'emergência' que você identifica não é só o roteador concentrando tarefas?"
5. "Quais são as implicações práticas concretas?"

Preparar respostas para cada uma antes da defesa.

---

## 7. Itens Concretos para a Próxima Sessão

1. Revisar e ajustar cada NOTA inserida nos capítulos
2. Decidir cada DECISÃO NARRATIVA listada acima
3. Criar `references.bib`
4. Aguardar conclusão dos sweeps (~6-7 horas)
5. Adicionar seção 4.6 com sweep results
6. Re-rodar `analyze_glmm.py --all` incluindo resultados de sweep
7. Escrever Resumo e Abstract
