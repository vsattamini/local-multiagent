# 1. INTRODUÇÃO

> 📝 **NOTAS GERAIS PARA REVISÃO DESTE CAPÍTULO:**
> - Capítulo escrito ANTES dos resultados completos do sweep — pode precisar ajustes finos após análise dos sweeps de 7B/3B.
> - Hipótese H4 (feedback verificável quebra o limiar) foi PARCIALMENTE REFUTADA. Considerar se manter como hipótese formal ou reescrever como "questão a investigar".
> - Verificar se todas as citações têm entrada no .bib (não criado ainda).
> - Tom geral: tentei evitar reivindicações infladas (alinhado com La Malfa et al. 2025). Verifique se algum trecho ainda soa exagerado.
> - Falta seção opcional: "Motivação Pessoal" / "Trajetória da Pesquisa" comum em ABNT UERJ — adicionar se exigido pelo programa.

## 1.1 Contexto

A última década testemunhou uma revolução nas capacidades de modelos de linguagem (Language Models, LMs), com modelos de centenas de bilhões de parâmetros apresentando comportamentos emergentes em tarefas de programação, raciocínio e geração de texto (Brown et al., 2020; OpenAI, 2023). Este progresso foi inicialmente atribuído ao paradigma de escala — a hipótese de que aumentar parâmetros, dados e computação produziria capacidades sucessivamente mais sofisticadas (Kaplan et al., 2020).

Entretanto, a partir de 2023, uma linha de pesquisa emergente demonstrou que **modelos pequenos** (Small Language Models, SLMs, com menos de 10 bilhões de parâmetros), quando treinados com dados de alta qualidade, podem rivalizar com modelos vastamente maiores em domínios específicos (Abdin et al., 2024). Este desenvolvimento abriu uma nova fronteira de pesquisa: a possibilidade de executar sistemas de inteligência artificial sofisticados em hardware de consumo, democratizando o acesso a capacidades antes restritas a centros de dados especializados.

Paralelamente, a comunidade de pesquisa explorou arquiteturas **multi-agente**, onde múltiplas instâncias de LMs colaboram para resolver tarefas complexas. Frameworks como MetaGPT (Hong et al., 2024), AutoGen (Wu et al., 2024) e ChatDev (Qian et al., 2024) demonstraram que coordenação estruturada entre agentes especializados pode amplificar a capacidade individual de cada modelo, com ChatDev gerando software funcional em menos de sete minutos.

A convergência destas duas linhas — SLMs eficientes e arquiteturas multi-agente — sugere uma direção promissora: **populações de SLMs coordenados** poderiam produzir sistemas de IA capazes a custos significativamente menores que LLMs frontier. Esta perspectiva motiva o argumento de Belcak e Heinrich (2025) de que SLMs são suficientes para a maioria das tarefas agênticas.

> 📝 **NOTA:** Belcak & Heinrich (2025) é position paper da NVIDIA — talvez vale citar com qualificação ("argumentam, em paper de posição..."). O paper é provocativo mas não tem evidência empírica forte. Considerar acrescentar contraposição imediata aqui ou deixar para o cap 2.

## 1.2 O Problema

Os benefícios da abordagem multi-agente com SLMs dependem criticamente da capacidade dos agentes de **diferenciarem-se funcionalmente** — desenvolver papéis especializados que, conjuntamente, cubram a diversidade de subtarefas necessárias. Em arquiteturas estabelecidas como MetaGPT, esta diferenciação é imposta explicitamente via prompts de persona ("você é um Gerente de Produto", "você é um Engenheiro").

Surge a questão: **populações de agentes idênticos podem desenvolver especialização emergente, sem atribuição explícita de papéis?**

Esta questão não é meramente acadêmica. Especialização emergente reduziria a necessidade de engenharia manual de prompts, aproximando sistemas multi-agente do paradigma de inteligência de enxame (swarm intelligence) — onde comportamento coletivo sofisticado emerge de interações locais simples (Bonabeau et al., 1999). Para SLMs, especialização emergente representaria um caminho para capacidade equivalente a modelos maiores, sem o custo computacional correspondente.

Entretanto, trabalhos recentes documentaram um **limiar de capacidade** abaixo do qual auto-organização em populações de LMs falha:

- **Riedl (2025)** demonstrou que Llama-3.1-8B, sem prompts de persona, alcançou apenas 10% de sucesso em uma tarefa de coordenação que modelos maiores resolvem com facilidade. A diferenciação observada foi caracterizada como "drift estocástico", não especialização estável.

- **Cemri et al. (2025)** documentaram taxas de falha entre 41% e 86.7% em sistemas multi-agente com LLMs em benchmarks estabelecidos, identificando 14 modos de falha distintos.

- **Dochkina (2026)**, em preprint subsequente, propôs que abaixo de aproximadamente 7 bilhões de parâmetros, auto-organização pode ativamente prejudicar o desempenho — modelos menores não possuiriam capacidade cognitiva suficiente para simultaneamente resolver a tarefa e coordenar com outros agentes.

> 📝 **NOTA CRÍTICA:** Dochkina (2026) deve ser citada com cautela — preprint MIPT, autor único, referencia "GPT-5.4" e "Gemini-3-flash" que não existem em abr/2026. Considere: (a) usar como evidência qualitativa apenas, ou (b) remover totalmente até peer review. A versão atual cita com qualificação no cap 2; aqui está mais leve. Decidir nível de cautela.

Estes achados levantam dúvidas fundamentais sobre a viabilidade de populações de SLMs auto-organizadas. **Mas a literatura existente apresenta uma lacuna importante**: os experimentos foram conduzidos com feedback escalar sintético (Riedl) ou julgamento por LLM (Dochkina), modalidades que adicionam ruído e ambiguidade ao sinal de aprendizado. Permanece em aberto se **feedback verificável** — testes unitários objetivos, com sinal binário não-ambíguo — poderia reduzir a carga cognitiva da coordenação e quebrar o limiar documentado.

## 1.3 Objetivos

### 1.3.1 Objetivo Geral

Investigar a existência, localização e mecanismos do limiar de capacidade para emergência de especialização funcional em populações de SLMs idênticos, utilizando feedback verificável de execução de testes.

### 1.3.2 Objetivos Específicos

1. **Caracterizar empiricamente** o limiar de capacidade através de experimentos comparativos com modelos de 1.5B, 3B e 7B parâmetros (Qwen2.5-Coder), mantendo arquitetura e dados de treinamento constantes.

2. **Identificar mecanismos moduladores** da emergência, investigando o efeito de temperatura de roteamento (controle de exploração vs. explotação) e tamanho de população (3 a 10 agentes).

3. **Validar métricas operacionalizadas** de especialização (S, D, F) adaptadas de medidas estabelecidas em ecologia (Theil, 1970; Blüthgen, 2006) e teoria da informação, fornecendo um framework reproduzível para futuras investigações.

4. **Avaliar criticamente** se feedback verificável (testes unitários do HumanEval) é suficiente para induzir especialização em modelos abaixo do limiar documentado por Riedl e proposto por Dochkina.

## 1.4 Questões de Pesquisa

A presente pesquisa investiga três questões de pesquisa centrais:

**RQ1: Existe um limiar de capacidade abaixo do qual a especialização emergente falha em populações de SLMs idênticos com feedback verificável?**

Esta questão testa empiricamente, em três escalas (1.5B, 3B, 7B), se há um ponto de transição onde a auto-organização emerge. Os resultados são interpretados à luz dos achados de Riedl (2025) e da hipótese de Dochkina (2026).

**RQ2: Quais mecanismos modulam a localização e a forma deste limiar?**

Esta questão investiga se intervenções no sistema de coordenação — temperatura do roteador, tamanho da população — podem deslocar ou contornar o limiar de capacidade. Identifica também se o trade-off entre especialização e desempenho varia com a escala do modelo.

**RQ3: Feedback verificável é uma condição suficiente para emergência em modelos abaixo do limiar?**

Esta questão compara os achados desta pesquisa com a literatura existente, avaliando se a substituição de feedback escalar/LLM-judge por feedback binário objetivo (testes unitários) é suficiente para induzir emergência em modelos menores.

## 1.5 Hipóteses

Com base na literatura revisada, formulamos as seguintes hipóteses:

- **H1**: Existe um limiar de capacidade entre 1.5B e 7B parâmetros, abaixo do qual a especialização não emerge na configuração de linha de base (esperado a partir de Riedl, 2025; Dochkina, 2026).

- **H2**: Intervenções no roteador (baixa temperatura) podem induzir especialização em modelos abaixo do limiar, mas com possível trade-off de desempenho.

- **H3**: Populações maiores aumentam a pressão competitiva por diferenciação, induzindo especialização mesmo em modelos menores.

- **H4**: Feedback verificável binário é mais eficaz que feedback escalar ou LLM-judge para induzir emergência, dado seu sinal não-ambíguo.

> 📝 **NOTA SOBRE HIPÓTESES:**
> - H1: CONFIRMADA pelos resultados (3B baseline também falha — limiar entre 3B e 7B)
> - H2: CONFIRMADA, com nuance importante (trade-off ocorre em 1.5B mas DESAPARECE em 3B)
> - H3: PARCIALMENTE CONFIRMADA (5 agentes ajuda 1.5B; ganho marginal em 3B)
> - H4: REFUTADA — feedback verificável NÃO é suficiente para quebrar o limiar
>
> Considerar: reescrever H4 como "Feedback verificável é necessário mas pode não ser suficiente" para alinhar com o que a evidência efetivamente mostrou. Alternativamente, manter H4 forte e celebrar a refutação como achado.

## 1.6 Contribuições

Esta dissertação oferece quatro contribuições principais à literatura:

1. **Mapeamento empírico do limiar de capacidade** em uma família coerente de modelos (Qwen2.5-Coder 1.5B/3B/7B), permitindo isolamento do efeito de escala de outros fatores arquiteturais.

2. **Caracterização do limiar como gradiente, não fronteira binária**: identificamos três regimes distintos (sub-limiar, limiar, supra-limiar) onde intervenções de coordenação têm efeitos qualitativamente diferentes.

3. **Identificação do desaparecimento do trade-off especialização-desempenho** em modelos próximos ao limiar (3B parâmetros), fornecendo orientação prática para construção de sistemas multi-agente em hardware de consumo.

4. **Framework reproduzível de métricas** (S, D, F) adaptado de medidas estabelecidas, com testes de significância via permutação e regressão logística com termos de interação, evitando reivindicações infladas criticadas por La Malfa et al. (2025).

> 📝 **NOTA:** Quarto ponto é o mais defensável academicamente. Os outros três dependem da força dos resultados empíricos. Considerar reordenar: pôr o framework metodológico primeiro, ou fundir (3) e (4).

## 1.7 Estrutura da Dissertação

Esta dissertação está organizada em seis capítulos:

- **Capítulo 1 (Introdução)**: Apresenta o contexto, problema, objetivos, questões de pesquisa e contribuições da pesquisa.

- **Capítulo 2 (Revisão da Literatura)**: Revisa a literatura sobre modelos de linguagem pequenos, sistemas multi-agente, emergência e auto-organização, o limiar de capacidade documentado, benchmarks de código, e contradições relevantes na literatura.

- **Capítulo 3 (Metodologia)**: Descreve a arquitetura do sistema (agentes, roteador de afinidade, executor de testes), as métricas de especialização (S, D, F) com suas raízes em ecologia e teoria da informação, e o design experimental.

- **Capítulo 4 (Resultados)**: Apresenta os resultados experimentais, organizados em três temas: o limiar de capacidade (1.5B, 3B, 7B), os mecanismos moduladores (temperatura, população), e a análise estatística (permutações, qui-quadrado, regressão logística).

- **Capítulo 5 (Discussão)**: Discute as implicações dos achados para a teoria de emergência em sistemas de LM, compara os resultados com a literatura (Riedl, Dochkina, Cemri), aborda limitações metodológicas e ameaças à validade.

- **Capítulo 6 (Conclusão)**: Sintetiza as contribuições, responde explicitamente às questões de pesquisa, e propõe direções para trabalhos futuros, incluindo extensão para benchmarks mais realistas (SWE-bench Verified) e investigação de feedback denso para modelos sub-limiar.
