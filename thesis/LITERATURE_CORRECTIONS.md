# Correções de Literatura - Abril 2026

Este documento lista correções obrigatórias para citações e framing na dissertação.

---

## 1. Correções de Citação

### 1.1 Choi et al. "Debate or Vote"

**ERRADO:**
```
OpenReview iUjGNJzrF1
```

**CORRETO:**
```
arXiv:2508.17536
NeurIPS 2025 Spotlight
```

**Localização:** cap2_revisao.md, seção 2.2.3

---

### 1.2 El Farol Bar Paper

**ERRADO:**
```
Iwasaki et al. (arXiv:2509.04537)
```

**CORRETO:**
```
Takata, Masumori, e Ikegami (arXiv:2509.04537)
```

**Localização:** cap2_revisao.md, seção 2.3.4

---

### 1.3 Dochkina - Reframing Necessário

**PROBLEMA:**
O preprint Dochkina (arXiv:2603.28990) é:
- Autor único, MIPT
- Sem peer review
- Referencia modelos que não existem ("GPT-5.4", "Gemini-3-flash")
- Effect sizes específicos não verificáveis

**ANTES:**
```markdown
Dochkina (arXiv:2603.28990, março 2026) conduziu experimento de larga escala...
O resultado surpreendente: **abaixo de um limiar de capacidade, auto-organização prejudica desempenho**
```

**DEPOIS:**
```markdown
Dochkina (arXiv:2603.28990) propõe, em preprint ainda não revisado por pares,
que abaixo de um limiar de capacidade, auto-organização pode prejudicar desempenho.
Embora os effect sizes específicos reportados (Cohen's d=1.86, +14%, +44%)
requeiram validação independente, a hipótese qualitativa é consistente com
achados de Riedl (2025) e Cemri et al. (2025).
```

**Localização:** cap2_revisao.md, seção 2.4.2

---

## 2. Citações Faltantes (Críticas)

### 2.1 Cemri et al. MAST (NeurIPS 2025 D&B)

**Referência:** arXiv:2503.13657

**Por que é crítico:**
- Taxonomia mais rigorosa de failure modes em MAS
- 14 modos de falha identificados
- 41-86.7% failure rates documentados
- Inclui Qwen2.5 family

**Onde adicionar:** Seção 2.2.4 (Limitações dos Sistemas Atuais)

---

### 2.2 MapCoder-Lite

**Referência:** arXiv:2509.17489

**Por que é crítico:**
- Demonstra colapso de 7B vanilla em multi-agent coding
- 13.2% vanilla → 28.3% com distilação
- Evidência direta do agent-level wall

**Onde adicionar:** Seção 2.4 (O Limiar de Capacidade)

---

### 2.3 Theil (1970) - Para Métricas

**Referência:** Theil, H. (1970). On the estimation of relationships involving qualitative variables. American Journal of Sociology, 76(1), 103-154.

**Por que é crítico:**
- Fonte original do coeficiente de incerteza
- Necessário para creditar S corretamente

**Onde adicionar:** Seção de métricas em cap3

---

### 2.4 Blüthgen et al. (2006) - Para Métricas

**Referência:** Blüthgen, N. et al. (2006). Measuring specialization in species interaction networks. BMC Ecology, 6(1), 1-12.

**Por que é crítico:**
- H₂' em ecologia = nosso S
- Validação de que a métrica é estabelecida

**Onde adicionar:** Seção de métricas em cap3

---

## 3. Contradições a Documentar

### 3.1 Adicionar Nova Seção em cap2

**Título sugerido:** 2.7 Contradições na Literatura

**Conteúdo:**

```markdown
## 2.7 Contradições na Literatura

A literatura sobre sistemas multi-agente com SLMs apresenta contradições
importantes que esta pesquisa deve engajar.

### 2.7.1 Otimismo vs. Falhas Empíricas

Belcak e Heinrich (2025) argumentam que SLMs são suficientes para tarefas
agênticas, citando reduções de custo de 10-30×. Entretanto, estudos empíricos
documentam taxas de falha de 41-86.7% (Cemri et al., 2025) e overhead de
coordenação de até 300× (Rahman & Schranz, 2025).

A reconciliação proposta: a eficácia de SLMs depende do tipo de tarefa.
Single-agent function calling com schema constraints é viável; multi-agent
coordination com contexto compartilhado não é.

### 2.7.2 Inversão de Rewards em Pequena Escala

A narrativa popular sugere que verifiable rewards (RLVR) fecham a lacuna
de modelos pequenos. Evidência recente (arXiv:2604.02621) inverte esta
expectativa: em modelos de 125M-350M, judge rewards superam verifiable
rewards por 5-10 pontos em raciocínio matemático.

Implicação: modelos pequenos precisam de densidade de sinal, não
necessariamente correção de sinal.
```

---

## 4. Checklist de Aplicação

### cap2_revisao.md

- [ ] Corrigir citação Choi (seção 2.2.3)
- [ ] Corrigir citação Takata (seção 2.3.4)
- [ ] Reframing Dochkina com caveats (seção 2.4.2)
- [ ] Adicionar Cemri MAST (seção 2.2.4)
- [ ] Adicionar MapCoder-Lite (seção 2.4)
- [ ] Adicionar seção 2.7 (Contradições)

### cap3_metodologia.md

- [ ] Adicionar citação Theil (1970) em 3.3.1
- [ ] Adicionar citação Blüthgen (2006) em 3.3.1
- [ ] Adicionar caveat sobre PID vs nossa abordagem simplificada

### cap4_resultados.md

- [ ] Preencher dados 3B (após experimento)
- [ ] Preencher mixed-effects (após implementação)

---

*Documento de correções - 2026-04-25*
