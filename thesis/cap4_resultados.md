# 4. RESULTADOS

> 📝 **NOTAS GERAIS PARA REVISÃO DESTE CAPÍTULO:**
> - **PENDENTE — incluir resultados dos sweeps (~32 experimentos)** quando concluídos. Local sugerido: nova seção 4.6 "Análise de Sensibilidade via Sweeps Paramétricos".
> - O capítulo passou por DUAS grandes revisões de narrativa:
>   - V1: "limiar entre 1.5B e 3B" (BASEADO EM exp_3b_model isolado — INCORRETO)
>   - V2: "limiar entre 3B e 7B; 3B é zona de intervenções gratuitas" (atual, baseado em todos 4 experimentos 3B)
> - Verificar se o teste LR de Razão de Verossimilhança (seção 4.4.2) está bem incorporado — ele é a análise estatisticamente mais rigorosa, mas falha em condições com separação perfeita.
> - Considerar adicionar gráfico (heat map agente×tipo de tarefa) para cada condição. ABNT permite figuras coloridas.
> - Decidir: relatar percentuais com 1 ou 2 casas decimais? Atualmente misto.
> - **Decisão narrativa importante**: quanto enfatizar a refutação parcial de H4 (feedback verificável não basta)? Pode ser apresentado como achado positivo ou como limitação.

## 4.1 Visão Geral dos Experimentos

Este capítulo apresenta os resultados experimentais que investigam o limiar de capacidade para emergência de especialização em populações de SLMs idênticos. Foram conduzidos nove experimentos utilizando o benchmark HumanEval (164 problemas de programação em Python), variando três fatores: tamanho do modelo (1.5B, 3B, 7B parâmetros), temperatura do roteador (0.1 a 0.5), e tamanho da população (3 a 5 agentes).

A Tabela 1 apresenta um resumo dos principais resultados.

| Experimento | Modelo | Agentes | Temp | Pass@1 | S | S sig? | Padrão |
|-------------|--------|---------|------|--------|------|--------|--------|
| Linha de base | 1.5B | 3 | 0.5 | 57.3% | 0.009 | Não | no_emergence |
| Baixa temp | 1.5B | 3 | 0.1 | 53.7% | 0.196 | Sim | gradual_drift |
| 5 agentes | 1.5B | 5 | 0.3 | 62.2% | 0.133 | Sim | gradual_drift |
| 3B baseline | 3B | 3 | 0.5 | 80.5% | 0.022 | **Não** | no_emergence |
| 3B temp 0.3 | 3B | 3 | 0.3 | 74.4% | 0.058 | Sim | no_emergence |
| 3B baixa temp | 3B | 3 | 0.1 | **81.1%** | **0.390** | Sim | gradual_drift |
| 3B 5 agentes | 3B | 5 | 0.3 | 79.9% | 0.115 | Sim | gradual_drift |
| Modelo 7B | 7B | 3 | 0.3 | 84.8% | 0.116 | Sim | gradual_drift |

*Nota: S = Índice de Especialização; sig = significância estatística (p < 0.05)*

O resultado mais marcante é que **tanto 1.5B quanto 3B falham na configuração de linha de base** (temperatura 0.5). A emergência de especialização não é automática em nenhum dos modelos menores — depende de intervenções como baixa temperatura ou população expandida. Apenas o modelo 7B apresenta emergência robusta sem intervenções.

---

## 4.2 O Limiar de Capacidade

O primeiro objetivo desta pesquisa foi investigar a existência de um limiar de capacidade abaixo do qual a especialização emergente falha. Os resultados confirmam esta hipótese.

### 4.2.1 Modelo 1.5B: Ausência de Emergência

Na configuração de linha de base com Qwen2.5-Coder-1.5B, três agentes idênticos e temperatura de roteamento 0.5, o sistema não apresentou especialização significativa. O Índice de Especialização observado (S = 0.009) não diferiu estatisticamente da distribuição nula gerada por 1000 permutações aleatórias (média nula = 0.016, σ = 0.009, p = 0.767).

A matriz de afinidade dos agentes revela desempenho uniforme entre tipos de tarefa:

| Agente | String | Math | List | Logic |
|--------|--------|------|------|-------|
| 0 | 0.62 | 0.53 | 0.56 | 0.75 |
| 1 | 0.50 | 0.67 | 0.60 | 1.00 |
| 2 | 0.42 | 0.44 | 0.56 | 1.00 |

As taxas de sucesso variam entre 0.42 e 0.67, sem padrão claro de especialização funcional.

### 4.2.2 Modelo 7B: Emergência Confirmada

Em contraste, o modelo Qwen2.5-Coder-7B apresentou especialização estatisticamente significativa (S = 0.116, p < 0.001). O teste qui-quadrado de diferenciação funcional também foi significativo (χ² = 65.0, p < 0.0001, V de Cramér = 0.48).

A distribuição de tarefas entre agentes mostra clara assimetria:

| Agente | Total de Tarefas | Taxa de Sucesso |
|--------|------------------|-----------------|
| 0 | 19 (12%) | 78.9% |
| 1 | 82 (50%) | 87.8% |
| 2 | 63 (38%) | 82.5% |

O Agente 1 emergiu como generalista dominante, enquanto os Agentes 0 e 2 se especializaram em subconjuntos de tarefas.

> 📝 **NOTA SEÇÃO 4.2.3:** Esta é a seção MAIS importante do capítulo. O achado de "três zonas" é o aporte conceitual original. Sugestões:
> - Considerar adicionar diagrama visual das três zonas
> - O termo "três zonas" pode ficar mais formal: "três regimes" (já usado em alguns lugares)
> - A descoberta de que 3B baseline também falha (não só 1.5B) é o twist principal — talvez começar a seção com isso para impacto

### 4.2.3 Modelo 3B: O Limiar é Condicional

Os experimentos com Qwen2.5-Coder-3B revelam um achado inesperado: **o modelo 3B também falha na configuração de linha de base**, similarmente ao 1.5B.

#### 4.2.3.1 Baseline 3B: Ausência de Emergência

Na configuração idêntica à linha de base do 1.5B (temperatura 0.5, 3 agentes), o modelo 3B não apresentou especialização significativa:

| Métrica | 1.5B baseline | 3B baseline |
|---------|---------------|-------------|
| S | 0.009 | 0.022 |
| S significativo | Não (p=0.767) | **Não (p=0.221)** |
| Pass@1 | 57.3% | 80.5% |
| χ² | 3.02 | 7.71 |
| F significativo | Não | Não |

Embora o 3B apresente desempenho absoluto muito superior (+23.2 pp em Pass@1), a estrutura de especialização não emerge. Os agentes distribuem tarefas de forma relativamente uniforme (53/59/52), sem diferenciação funcional.

**Resultado principal:** O limiar de capacidade para emergência **não assistida** está localizado **entre 3B e 7B parâmetros**.

#### 4.2.3.2 Intervenções no 3B: Emergência Sem Trade-offs

A diferença crítica entre 1.5B e 3B emerge quando aplicamos intervenções:

| Condição | 1.5B S | 1.5B Pass@1 | 3B S | 3B Pass@1 |
|----------|--------|-------------|------|-----------|
| Baseline (temp 0.5) | 0.009 | 57.3% | 0.022 | 80.5% |
| Baixa temp (0.1) | 0.196 | 53.7% (-3.6pp) | **0.390** | **81.1% (+0.6pp)** |
| 5 agentes | 0.133 | 62.2% (+4.9pp) | 0.115 | 79.9% (-0.6pp) |

**Achado central:** No 1.5B, baixa temperatura força especialização às custas de desempenho. No 3B, **baixa temperatura amplifica especialização sem degradar desempenho** — o trade-off desaparece.

O experimento 3B baixa temp produziu:
- S = 0.390 (o maior observado em qualquer experimento)
- Pass@1 = 81.1% (superior ao baseline 3B)
- χ² = 115.9, V = 0.66 (efeito muito grande)

A distribuição de tarefas mostra especialização genuína:

| Agente | Tarefas | Sucesso | Especialização |
|--------|---------|---------|----------------|
| 0 | 13 (8%) | 76.9% | List (87.5%) |
| 1 | 45 (27%) | 77.8% | List (83%), Logic (80%) |
| 2 | 106 (65%) | 83.0% | String (84%), Math (82%) |

O Agente 2 emergiu como generalista dominante em string/math, enquanto Agentes 0 e 1 se especializaram em list/logic.

#### 4.2.3.3 Implicações para o Limiar de Capacidade

Os resultados sugerem que o "limiar de capacidade" não é um ponto único, mas um **gradiente com três zonas**:

| Zona | Modelo | Emergência | Trade-off |
|------|--------|------------|-----------|
| **Sub-limiar** | 1.5B | Requer intervenção | Intervenções degradam desempenho |
| **Limiar** | 3B | Requer intervenção | **Intervenções não degradam** |
| **Supra-limiar** | 7B | Espontânea | Intervenções opcionais |

**Implicação para acessibilidade:** O modelo 3B representa a "fronteira de acessibilidade" — o menor modelo onde intervenções produzem especialização sem trade-offs. Em GPUs de consumo (6GB VRAM), configurações com baixa temperatura permitem sistemas multi-agente eficazes.

---

## 4.3 Mecanismos Moduladores da Emergência

Além do tamanho do modelo, investigamos dois mecanismos que modulam a emergência de especialização: temperatura do roteador e tamanho da população.

### 4.3.1 Efeito da Temperatura do Roteador

A temperatura do roteador controla o balanço entre exploração (distribuição uniforme de tarefas) e explotação (concentração em agentes com melhor histórico). Comparamos temperaturas de 0.1 (baixa) versus 0.5 (linha de base) em ambos os modelos 1.5B e 3B.

| Modelo | Temp | S | Pass@1 | Distribuição de Tarefas |
|--------|------|------|--------|------------------------|
| 1.5B | 0.5 | 0.009 | 57.3% | Uniforme (65/59/40) |
| 1.5B | 0.1 | 0.196 | 53.7% | Concentrada (127/21/16) |
| 3B | 0.5 | 0.022 | 80.5% | Uniforme (53/59/52) |
| 3B | 0.1 | **0.390** | **81.1%** | Concentrada (13/45/106) |

O resultado é dramático: no 3B, a temperatura baixa induziu o **maior índice de especialização observado** em todos os experimentos (S = 0.390, p < 0.001), com Pass@1 mantido em níveis altos (81.1%).

**Padrões qualitativamente distintos entre 1.5B e 3B:**

- **1.5B baixa temp:** O Agente 0 acumulou 77% das tarefas (127 de 164), tornando-se um generalista dominante. A matriz de afinidade revela desempenho uniforme entre tipos (0.53-0.56) — **especialização degenerada**.

- **3B baixa temp:** O Agente 2 acumulou 65% das tarefas, mas com **especialização funcional clara**: domina string (84%) e math (82%), enquanto Agentes 0 e 1 se especializam em list (83-87%) e logic (50-80%). Os agentes desenvolvem perfis funcionais distintos, não apenas concentração de carga.

A diferença qualitativa sugere que a baixa temperatura não força especialização per se — ela apenas concentra tarefas. Se o modelo possui capacidade suficiente, a concentração se traduz em diferenciação funcional. Caso contrário, resulta em monopolização sem diferenciação.

### 4.3.2 Efeito do Tamanho da População

A segunda intervenção testou se populações maiores aumentam a pressão competitiva e facilitam a especialização. Comparamos 5 agentes versus 3 agentes em ambos os modelos 1.5B e 3B.

| Modelo | Agentes | Temp | S | Pass@1 |
|--------|---------|------|------|--------|
| 1.5B | 3 | 0.5 | 0.009 | 57.3% |
| 1.5B | 5 | 0.3 | 0.133 | 62.2% |
| 3B | 3 | 0.5 | 0.022 | 80.5% |
| 3B | 5 | 0.3 | 0.115 | 79.9% |

A população expandida produz especialização significativa em ambos os modelos (S = 0.133 e 0.115). O efeito sobre desempenho é positivo no 1.5B (+4.9pp) e neutro no 3B (-0.6pp), provavelmente devido ao efeito de teto (3B já apresenta desempenho elevado).

**Distribuição no 3B com 5 agentes:**

| Agente | Tarefas | Sucesso | Perfil |
|--------|---------|---------|--------|
| 0 | 20 (12%) | 80.0% | Especialista em list (92%) |
| 1 | 51 (31%) | 80.4% | Generalista (string 90%, math 79%, list 89%) |
| 2 | 29 (18%) | 75.9% | Especialista em logic e list (100%) |
| 3 | 37 (23%) | 75.7% | Especialista em math (78%) |
| 4 | 27 (16%) | 88.9% | Especialista em string (92%) e logic (100%) |

A população expandida produz especialização funcional clara, com cada agente desenvolvendo um perfil distinto. Comparativamente, no 1.5B com 5 agentes a hierarquia é mais marcada por desempenho desigual (42-72%) do que por diferenciação funcional. O 3B mantém desempenho uniformemente alto (75-89%) entre todos os agentes, com diferenciação por especialização.

### 4.3.3 Interação entre Mecanismos: O Efeito do Tamanho do Modelo

A Tabela 2 sintetiza os efeitos das intervenções em cada escala de modelo:

| Intervenção | 1.5B ΔS | 1.5B ΔPass@1 | 3B ΔS | 3B ΔPass@1 |
|-------------|---------|--------------|-------|------------|
| Baixa temp (0.1 vs baseline) | +0.187 | **-3.6 pp** | +0.368 | **+0.6 pp** |
| 5 agentes (vs 3) | +0.124 | +4.9 pp | +0.093 | -0.6 pp |

**Achado crítico:** O trade-off entre especialização e desempenho é **específico do tamanho do modelo**:

- **1.5B:** Baixa temperatura força especialização mas degrada desempenho (trade-off negativo)
- **3B:** Baixa temperatura amplifica especialização sem degradar desempenho (trade-off neutro/positivo)
- **7B:** Emergência espontânea, intervenções opcionais

Isto sugere que o "limiar de capacidade" não é apenas sobre habilitar emergência, mas sobre **eliminar trade-offs**. Abaixo de 3B, intervenções têm custos. A partir de 3B, intervenções são "gratuitas".

A explicação proposta: modelos menores dedicam capacidade cognitiva excessiva ao processamento do contexto acumulado. A baixa temperatura concentra tarefas em poucos agentes, reduzindo a diversidade de contexto mas sobrecarregando agentes individuais. No 3B, a capacidade extra permite absorver esta carga sem degradação.

---

## 4.4 Análise Estatística

### 4.4.1 Validação do Índice de Especialização

Para cada experimento, a significância estatística de S foi avaliada comparando o valor observado contra uma distribuição nula gerada por 1000 permutações aleatórias das atribuições agente-tarefa. A Tabela 3 apresenta os resultados completos.

| Experimento | S obs. | Média nula | σ nulo | p-valor | Significativo |
|-------------|--------|------------|--------|---------|---------------|
| 1.5B baseline | 0.009 | 0.016 | 0.009 | 0.767 | Não |
| 1.5B baixa temp | 0.196 | 0.017 | 0.010 | <0.001 | Sim |
| 1.5B 5 agentes | 0.133 | 0.032 | 0.013 | <0.001 | Sim |
| **3B baseline** | **0.022** | **0.016** | **0.009** | **0.221** | **Não** |
| 3B temp 0.3 | 0.058 | 0.016 | 0.009 | <0.001 | Sim |
| **3B baixa temp** | **0.390** | **0.017** | **0.009** | **<0.001** | **Sim** |
| 3B 5 agentes | 0.115 | 0.033 | 0.013 | <0.001 | Sim |
| 7B | 0.116 | 0.016 | 0.009 | <0.001 | Sim |

As condições de linha de base (1.5B e 3B na temperatura 0.5) são as únicas cujo S observado não supera a distribuição nula, confirmando ausência de emergência espontânea em ambos os modelos pequenos. Nas demais condições, os valores de S excedem o percentil 99.9 da distribuição nula.

Notavelmente, o experimento **3B baixa temp produziu S = 0.390**, o maior valor observado em todos os experimentos com modelos pequenos — superando inclusive o modelo 7B em configuração padrão (S = 0.116). Isto confirma que intervenções no roteamento podem produzir especialização em modelos menores quando o modelo possui capacidade base suficiente para aproveitá-las.

### 4.4.2 Diferenciação Funcional: Teste de Razão de Verossimilhança

Para avaliar se os agentes apresentam desempenho diferenciado por tipo de tarefa, empregamos regressão logística com termo de interação `tipo_tarefa × agente`, comparando contra um modelo aditivo via teste de razão de verossimilhança (LR test).

**Nota sobre o design experimental:** O framework GLMM com efeito aleatório `(1|problema)` proposto por Riedl (2025) requer que cada problema seja tentado por múltiplos agentes. No design de roteamento por afinidade desta pesquisa, cada problema é resolvido por exatamente um agente (n_problemas = n_observações = 164), tornando o efeito aleatório inidentificável. Empregamos portanto o modelo de efeitos fixos, que permanece válido para o teste de interação:

```
H₀: P(sucesso) ~ tipo_tarefa + agente
H₁: P(sucesso) ~ tipo_tarefa × agente
LR: -2(ℓ₀ - ℓ₁) ~ χ²(df_full - df_null)
```

A Tabela 4 apresenta dois testes complementares: o teste qui-quadrado de contingência (sobre contagens de sucesso por célula tipo:agente) e o teste de razão de verossimilhança sobre o modelo logit.

| Experimento | χ² contig. | p (contig.) | V | LR χ² | p (LR) | Significativo (LR) |
|-------------|-----------|-------------|------|--------|---------|--------------------|
| 1.5B baseline | 3.02 | 0.806 | 0.13 | 3.70 | 0.718 | Não |
| 1.5B baixa temp | 69.80 | <0.001 | 0.63 | — | — | Separação¹ |
| 1.5B 5 agentes | 36.90 | <0.001 | 0.35 | — | — | Separação¹ |
| 3B baseline | 7.71 | 0.260 | 0.17 | 4.77 | 0.574 | Não |
| 3B temp 0.3 | 19.12 | 0.004 | 0.28 | 4.22 | 0.518 | **Não²** |
| 3B baixa temp | 115.90 | <0.001 | 0.66 | — | — | Separação¹ |
| 3B 5 agentes | 42.67 | <0.001 | 0.33 | — | — | Separação¹ |
| 7B | 65.01 | <0.001 | 0.48 | 15.60 | 0.008 | **Sim** |

*¹ Separação perfeita: distribuição de tarefas excessivamente concentrada (e.g., um agente com 0 sucessos em uma categoria) impede ajuste do modelo logit. Nestes casos, o teste qui-quadrado é o único disponível.*

*² O teste qui-quadrado mostra significância porque a distribuição de tarefas entre agentes é desigual; o teste LR (que controla pelos totais marginais) revela que as taxas de sucesso por categoria não diferem significativamente entre agentes — a "especialização" detectada é principalmente concentração de roteamento, não diferenciação funcional genuína.*

**Achado importante:** Apenas o **modelo 7B** apresenta diferenciação funcional **genuína** detectada pelo teste LR (χ² = 15.60, p = 0.008). Esta é uma descoberta crítica que refina a interpretação dos resultados anteriores: a significância do qui-quadrado em outros experimentos pode refletir padrões de roteamento (concentração) sem implicar especialização funcional substantiva.

> 📝 **NOTA CRÍTICA:** Este é um achado que TENSIONA com afirmações em outras partes do capítulo. Trechos que falam de "especialização emergente" no 3B com baixa temperatura (S=0.390) precisam ser RECONCILIADOS com este resultado:
> - Possibilidade 1: A separação perfeita IMPEDE o teste LR — a especialização pode ser real mas indetectável formalmente. Argumento: V de Cramér = 0.66 indica efeito grande SE a interação fosse identificável.
> - Possibilidade 2: A "especialização" no 3B com baixa temperatura é principalmente concentração de carga, similar ao 1.5B com baixa temperatura.
> Decidir como apresentar este trade-off interpretativo. Talvez vale uma seção dedicada "4.4.4 Sobre a Interpretação de Significância nos Casos com Separação".

Para os experimentos com separação perfeita (1.5B baixa temp, 1.5B 5 agentes, 3B baixa temp, 3B 5 agentes), a alta concentração de tarefas torna o teste LR não computável. Nestes casos, o efeito V de Cramér ≥ 0.33 indica que **se** a interação fosse identificável, seu tamanho seria substancial — porém o teste formal não está disponível.

### 4.4.3 Divergência de Contexto

A divergência de contexto (D) mede a dissimilaridade entre os exemplos acumulados por cada agente. Valores altos indicam que agentes desenvolveram "experiências" distintas.

| Experimento | D | S | Interpretação |
|-------------|------|------|---------------|
| 1.5B baseline | 0.741 | 0.009 | Alta divergência, sem especialização funcional |
| 1.5B baixa temp | 0.503 | 0.196 | Divergência moderada, especialização degenerada |
| 1.5B 5 agentes | 0.622 | 0.133 | Divergência substancial, especialização saudável |
| 3B baseline | 0.668 | 0.022 | Alta divergência, sem especialização funcional |
| 3B baixa temp | 0.836 | **0.390** | **Maior divergência, maior especialização** |
| 3B 5 agentes | 0.612 | 0.115 | Divergência substancial, especialização saudável |
| 7B | 0.648 | 0.116 | Divergência substancial, especialização saudável |

Nota-se que as configurações de linha de base (1.5B e 3B em temp 0.5) apresentam alta divergência (D = 0.741 e 0.668 respectivamente), apesar de não exibirem especialização significativa. Isto indica que **divergência de contexto é necessária mas não suficiente** para especialização funcional — os agentes acumularam exemplos diferentes, mas isso não se traduziu em diferenciação de desempenho.

Por outro lado, a configuração 3B baixa temp combina **maior divergência (D = 0.836) com maior especialização (S = 0.390)**, sugerindo que quando o modelo possui capacidade suficiente, divergência de contexto traduz-se diretamente em diferenciação funcional.

---

> 📝 **NOTA:** A seção 4.5 sintetiza tudo. Verificar se está consistente após inclusão dos sweeps (pendente). Em particular, se algum sweep mostrar comportamento anômalo (e.g., 7B com população grande quebra), ajustar conclusões.

## 4.5 Síntese dos Resultados

Os experimentos respondem às três questões de pesquisa, com nuances importantes que emergem da comparação completa entre 1.5B, 3B e 7B.

**RQ1: Existe um limiar de capacidade para emergência?**

Sim, mas o limiar é **mais sutil do que originalmente hipotetizado**. Existem três zonas distintas:

| Zona | Modelos | Característica |
|------|---------|----------------|
| **Sub-limiar** | 1.5B | Emergência possível com intervenções, mas com trade-off de desempenho |
| **Limiar** | 3B | Emergência possível com intervenções, **sem trade-off** |
| **Supra-limiar** | 7B | Emergência espontânea, intervenções opcionais |

Em condição idêntica de linha de base (temperatura 0.5), tanto 1.5B (S = 0.009) quanto 3B (S = 0.022) falham em produzir especialização significativa. Apenas o 7B (S = 0.116) apresenta emergência espontânea.

**RQ2: Quais mecanismos modulam a emergência?**

Três mecanismos foram identificados, com efeitos que dependem do tamanho do modelo:

- **Capacidade do modelo**: Pré-requisito para emergência espontânea (manifesta apenas a partir de 7B)
- **Temperatura baixa do roteador**: Mecanismo mais potente; produz S = 0.390 no 3B (maior valor observado), mas com trade-off negativo no 1.5B
- **Tamanho da população**: Efeito moderado e similar entre 1.5B e 3B (S ≈ 0.12-0.13)

O achado mais importante é que **o trade-off entre especialização e desempenho desaparece a partir de 3B parâmetros**. No 1.5B, baixa temperatura produz especialização degenerada (-3.6pp em Pass@1). No 3B, a mesma intervenção produz especialização saudável (+0.6pp em Pass@1).

**RQ3: Feedback verificável possibilita emergência?**

Confirmado com ressalvas importantes. O feedback binário (teste passou/falhou) do HumanEval foi suficiente para induzir especialização em todas as condições com modelos ≥ 3B, exceto em linha de base (temp 0.5). Entretanto:

- O 1.5B só apresentou emergência com intervenções fortes, e com trade-off de desempenho
- O 3B baseline também falhou — mostrando que feedback verificável **sozinho não é suficiente** abaixo do limiar
- A combinação capacidade adequada + feedback verificável + intervenção apropriada produz emergência saudável

Estes achados são consistentes com as previsões qualitativas de Riedl (2025) e a hipótese de Dochkina (2026), mas sugerem uma reformulação: o limiar não é absoluto, mas **gradiente**, e seus efeitos dependem da configuração do roteador.

**Implicação prática:** O modelo de 3B parâmetros representa a "fronteira de acessibilidade" — o menor modelo onde intervenções no roteamento produzem especialização sem trade-offs. Em GPUs de consumo com 6GB de VRAM, configurações com baixa temperatura permitem sistemas multi-agente eficazes.

**Implicação teórica:** O conceito de "limiar de capacidade" deve ser refinado de uma fronteira binária para um gradiente com três regimes: incapacidade total, capacidade condicional, e capacidade espontânea.
