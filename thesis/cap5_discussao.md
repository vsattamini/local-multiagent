# 5. DISCUSSÃO

> 📝 **NOTAS GERAIS PARA REVISÃO DESTE CAPÍTULO:**
> - Capítulo escrito em uma sentada — está mais "polido" mas também mais sujeito a sobreargumentação. Verifique se há reivindicações que excedem a evidência.
> - As três zonas (sub-limiar / limiar / supra-limiar) são o aporte conceitual ORIGINAL desta dissertação. Isso é forte? Ou só um descritivo dos dados? Pensar se justifica defesa em banca.
> - A comparação com Riedl/Dochkina/MapCoder está OK mas pode ser expandida com tabelas comparativas.
> - A seção 5.5 (Limitações) é boa mas pode ser mais autocrítica:
>   - Single-seed por configuração (mencionar mais explicitamente)
>   - Categorização manual de tarefas (4 categorias para 164 problemas — pequena!)
>   - Apenas Qwen2.5-Coder (não testou outras famílias)
>   - HumanEval pode estar em dados de treino do modelo
> - Falta seção opcional sobre **Implicações Éticas** — pode não ser necessário para o tema, mas alguns programas exigem.
> - Considerar se "Síntese" (5.6) é redundante com o Cap 6 (Conclusão).

## 5.1 Interpretação dos Achados Principais

Os experimentos conduzidos respondem afirmativamente à questão da existência de um limiar de capacidade, mas refinam substancialmente sua caracterização. A interpretação tradicional de um limiar binário — modelos abaixo de certo tamanho não exibem emergência, modelos acima sim — é uma simplificação inadequada. Os dados sugerem um **gradiente com três regimes** qualitativamente distintos.

### 5.1.1 O Modelo de Três Zonas

A configuração de linha de base (temperatura 0.5, três agentes) revelou um padrão inesperado: tanto o modelo 1.5B (S = 0.009, p = 0.767) quanto o modelo 3B (S = 0.022, p = 0.221) falharam em produzir especialização significativa. Apenas o modelo 7B (S = 0.116, p < 0.001) apresentou emergência espontânea.

Em contraste, quando intervenções foram aplicadas (baixa temperatura ou população expandida), padrões qualitativamente diferentes emergiram em cada modelo:

- **1.5B com intervenção** apresentou especialização ao custo de desempenho (Pass@1 cai de 57.3% para 53.7%). O Agente 0 monopolizou 77% das tarefas, sem desenvolver perfil funcional distinto.

- **3B com intervenção** apresentou especialização **sem custo de desempenho** (Pass@1 sobe de 80.5% para 81.1%). Os agentes desenvolveram perfis funcionais distintos: Agente 2 dominou string/math, Agentes 0 e 1 especializaram-se em list/logic.

- **7B sem intervenção** apresentou emergência saudável, com diferenciação funcional confirmada pelo teste de razão de verossimilhança (χ² = 15.60, p = 0.008).

Esta gradação sugere a seguinte interpretação teórica: o limiar de capacidade não governa apenas a possibilidade de emergência, mas a **qualidade** da especialização produzida. Abaixo do limiar, intervenções forçam concentração de roteamento sem diferenciação genuína. No limiar, intervenções permitem diferenciação verdadeira. Acima do limiar, diferenciação emerge espontaneamente.

> 📝 **NOTA:** A próxima sub-seção (5.1.2) é o achado metodológico mais importante. Verificar se a explicação está acessível para um leitor que não tem profundidade em estatística. Talvez vale um exemplo numérico simples para ilustrar a diferença entre os dois testes.

### 5.1.2 Concentração de Roteamento vs. Diferenciação Genuína

O teste de razão de verossimilhança revelou uma distinção crítica que o teste qui-quadrado tradicional obscurece. Em condições com forte intervenção (3B baixa temperatura, V de Cramér = 0.66), o qui-quadrado captura padrões dramáticos de assimetria nas contagens de tarefas. Entretanto, este efeito mistura duas fontes de variação:

1. **Variação na alocação de tarefas** (qual agente recebe qual tipo)
2. **Variação na taxa de sucesso** (quão bem cada agente performa em cada tipo, condicional à atribuição)

O teste LR isola a segunda — a única que constitui especialização funcional substantiva. Apenas o modelo 7B passou neste teste mais rigoroso. Esta descoberta tem implicações importantes para a literatura: trabalhos que reportam "especialização" baseando-se exclusivamente em distribuições de tarefas podem estar capturando artefatos de roteamento, não emergência cognitiva.

## 5.2 Comparação com a Literatura

### 5.2.1 Diálogo com Riedl (2025)

Os achados desta pesquisa são amplamente consistentes com Riedl (2025), mas com nuances importantes:

| Aspecto | Riedl (2025) | Esta pesquisa |
|---------|--------------|---------------|
| Modelo testado | Llama-3.1-8B | Qwen2.5-Coder 1.5B/3B/7B |
| Feedback | Escalar sintético | Binário verificável (testes) |
| Resultado linha de base | 10% sucesso, drift estocástico | 3B/1.5B falham; 7B sucede |
| Localização do limiar | ~8B | Entre 3B e 7B (consistente) |

A localização aproximada do limiar é congruente entre os trabalhos. Entretanto, a presente pesquisa adiciona evidência de que **feedback verificável não é suficiente** para quebrar o limiar em modelos sub-limiar — uma hipótese que motivou o desenho experimental e foi parcialmente refutada pelos resultados.

> 📝 **NOTA:** Manter a cautela com Dochkina. Atual versão equilibra "suporte qualitativo" com "preprint a verificar". Se ela tiver sido revisada/atualizada quando da defesa, atualizar este trecho.

### 5.2.2 Diálogo com Dochkina (2026)

A hipótese de Dochkina — de que abaixo de aproximadamente 7B parâmetros, auto-organização ativamente prejudica desempenho — recebe apoio qualitativo dos resultados, mas com importante refinamento:

- **Suporte**: O modelo 1.5B com baixa temperatura apresenta trade-off negativo (S sobe, Pass@1 cai), consistente com a hipótese de sobrecarga cognitiva.

- **Refinamento**: No modelo 3B, o trade-off **desaparece**. Intervenções de coordenação produzem especialização sem prejudicar desempenho. Isto sugere que o limiar de Dochkina pode ser mais baixo do que originalmente proposto (possivelmente 3B-7B em vez de >7B).

A hipótese de Dochkina deve ser citada com a cautela apropriada para um preprint de autor único que referencia modelos não-existentes em sua data de publicação. Os achados qualitativos (existência de trade-off em modelos pequenos) são consistentes com evidência independente; os números específicos requerem validação.

### 5.2.3 Diálogo com Cemri et al. (2025) e MapCoder-Lite

Cemri et al. (2025) documentaram taxas de falha de 41-86.7% em sistemas multi-agente com LLMs. Os resultados desta pesquisa são consistentes: o sistema baseado em SLM 1.5B em configuração padrão apresenta especialização nula (S = 0.009), embora atinja Pass@1 de 57.3% no benchmark.

MapCoder-Lite (arXiv:2509.17489) reportou que modelos 7B sem intervenção colapsam em 13.2% de sucesso em coordenação multi-agente, alcançando 28.3% após destilação direcionada. A presente pesquisa observa o oposto: o modelo Qwen2.5-Coder-7B atinge 84.8% Pass@1 com especialização significativa. A discrepância sugere que a tarefa específica e o tipo de coordenação afetam dramaticamente o limiar — coordenação via roteamento por afinidade é mais permissiva que coordenação via debate ou planejamento explícito.

## 5.3 Implicações Teóricas

### 5.3.1 Repensando o Conceito de Limiar de Capacidade

A literatura tem tratado o limiar de capacidade como uma fronteira binária: modelos têm ou não capacidade suficiente. Os achados desta pesquisa sugerem refinamento conceitual: o limiar é melhor caracterizado por uma **função de custo** que define quão custoso é induzir emergência em cada escala.

| Zona | Custo de emergência | Característica |
|------|---------------------|----------------|
| Sub-limiar (1.5B) | Alto | Intervenções degradam desempenho |
| Limiar (3B) | Moderado | Intervenções funcionam sem custo |
| Supra-limiar (7B) | Baixo | Emergência espontânea |

Esta perspectiva é alinhada com a teoria de inteligência de enxame (Bonabeau et al., 1999), que enfatiza o papel de mecanismos de feedback positivo e negativo. No regime sub-limiar, o feedback positivo (concentração via afinidade) não é suficientemente compensado pelo feedback negativo (saturação por carga cognitiva), resultando em colapso para um único agente dominante.

### 5.3.2 Aprendizado In-Context como Mecanismo Identificado

Os experimentos identificam o mecanismo causal proposto na literatura (Brown et al., 2020; Agarwal et al., 2024): o aprendizado in-context (ICL) através de exemplos acumulados é o veículo da diferenciação. Isto é evidenciado pela divergência de contexto (D) substancial em todas as condições (D ≥ 0.50), incluindo aquelas sem especialização funcional.

A descoberta de que **alta divergência de contexto não implica especialização** (1.5B baseline tem D = 0.741 mas S = 0.009) é teoricamente importante. Refuta a hipótese ingênua de que diferenças contextuais traduzem-se diretamente em diferenciação comportamental. A capacidade do modelo de **explorar** efetivamente o contexto é o gargalo.

## 5.4 Implicações Práticas

### 5.4.1 Acessibilidade

O modelo 3B representa uma fronteira crítica para sistemas multi-agente acessíveis. Em GPUs de consumo com 6GB de VRAM (e.g., RTX 3060), o Qwen2.5-Coder-3B em quantização Q4_K_M ocupa aproximadamente 2GB, deixando espaço para contexto e múltiplas instâncias. As intervenções identificadas (baixa temperatura, população moderada) são gratuitas computacionalmente.

Para profissionais e pesquisadores com restrições de hardware, esta pesquisa identifica configurações específicas que produzem especialização emergente sem trade-off de desempenho:

- **Configuração recomendada**: 3B parâmetros, 3-5 agentes, temperatura de roteador entre 0.1 e 0.3
- **Configurações a evitar**: 1.5B com intervenções fortes (especialização degenerada), temperaturas altas (sem emergência)

### 5.4.2 Limites do Argumento Pro-SLM

O argumento de Belcak e Heinrich (2025) — de que SLMs são suficientes para tarefas agênticas — recebe suporte parcial e qualificado dos resultados. Para tarefas resolvíveis pelo modelo individual, SLMs são viáveis. Para tarefas que requerem coordenação genuína entre agentes especializados, modelos abaixo de 7B apresentam limitações significativas, mesmo com intervenções de roteamento.

A reconciliação proposta: a viabilidade de SLMs depende do **tipo de coordenação** requerido. Roteamento por afinidade com tarefas independentes é viável a partir de 3B. Coordenação via debate, contexto compartilhado, ou planejamento hierárquico provavelmente requer modelos maiores.

## 5.5 Limitações e Ameaças à Validade

### 5.5.1 Limitações do Design Experimental

**Single-seed por configuração**: Cada experimento foi conduzido com uma única semente aleatória (42), por restrições de tempo computacional. Isto limita a estimativa de variância entre execuções. Trabalhos futuros devem replicar com múltiplas sementes (5-10) para obter intervalos de confiança robustos.

**Roteamento determinístico por afinidade**: Cada problema é atribuído a exatamente um agente. Isto impossibilita análise de efeitos mistos com (1|problema), exigindo o uso de regressão logística simples. Designs com múltiplas tentativas por problema (ensemble) permitiriam análise GLMM completa.

**Benchmark único**: HumanEval, embora estabelecido, apresenta limitações documentadas (Liu et al., 2023): cobertura limitada de testes (corrigida em HumanEval+) e potencial contaminação de dados de treinamento. Generalização para SWE-bench Verified ou Terminal-Bench é tema para trabalho futuro.

### 5.5.2 Ameaças à Validade

**Validade interna**: A categorização de tarefas em quatro tipos (string, math, list, logic) foi conduzida manualmente. Embora consistente com convenções do campo, é parcialmente subjetiva. Análises de sensibilidade com categorizações alternativas seriam valiosas.

**Validade externa**: Os achados são específicos para a família de modelos Qwen2.5-Coder. Modelos com diferentes objetivos de pré-treinamento (e.g., Phi-3, Llama-3) podem apresentar limiares deslocados. A generalização requer replicação cross-modelo.

**Validade construto**: A medida S (Índice de Especialização) é uma adaptação do coeficiente de incerteza de Theil (1970), que assume independência entre observações. A correlação intra-agente induzida pelo roteamento por afinidade pode inflar S. O teste de permutação aborda parcialmente esta preocupação, mas análises com correção para autocorrelação seriam mais rigorosas.

> 📝 **NOTA:** Esta sub-seção (5.5.3) é a mais "ousada" do capítulo — explicitamente recusa reivindicações comuns na literatura. Considerar:
> - É bom — mostra rigor metodológico
> - Pode soar arrogante ou defensivo. Suavizar com "Em conformidade com a crítica de La Malfa..." (já feito)
> - Verificar se as reivindicações listadas como NÃO-suportadas são, de fato, evitadas em todos os capítulos. Em particular: cap 6 (Conclusão) usa termos como "diferenciação funcional genuína" — pode ser interpretado como reivindicação inflada por leitor adversário.

### 5.5.3 Reivindicações Cuidadosamente Calibradas

Em conformidade com a crítica metodológica de La Malfa et al. (2025), evitamos as seguintes reivindicações que a evidência não suporta:

- ❌ "Demonstração de inteligência coletiva emergente"
- ❌ "Comportamento similar a humano em coordenação"
- ❌ "Solução geral para limitações de LLMs pequenos"

As reivindicações suportadas são mais modestas:

- ✓ Existe um limiar de capacidade para emergência de especialização
- ✓ O limiar é mais bem caracterizado como gradiente
- ✓ Intervenções específicas modulam a localização efetiva do limiar
- ✓ Feedback verificável é necessário, mas não suficiente, para quebrar o limiar

## 5.6 Síntese

A presente pesquisa contribui para o entendimento do limiar de capacidade em três dimensões: refina sua caracterização (de fronteira para gradiente), identifica mecanismos moduladores (temperatura, população), e distingue concentração de roteamento de diferenciação funcional genuína. Os resultados são amplamente consistentes com a literatura existente, estendendo-a com o caso específico de feedback verificável no domínio de geração de código.

A principal lição prática é a identificação do modelo 3B como ponto de inflexão para acessibilidade — o menor tamanho onde sistemas multi-agente especializados são viáveis em hardware de consumo, sem trade-off de desempenho. A principal lição teórica é a necessidade de testes estatísticos mais rigorosos (LR sobre logit) que distinguam concentração de diferenciação substantiva.
