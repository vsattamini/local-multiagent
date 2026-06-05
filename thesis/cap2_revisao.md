# 2. REVISÃO DA LITERATURA

> 📝 **NOTAS GERAIS PARA REVISÃO DESTE CAPÍTULO:**
> - Citações já corrigidas em pass anterior (ver LITERATURE_CORRECTIONS.md). Verificar:
>   - Choi et al. → arXiv:2508.17536 (NeurIPS 2025 Spotlight, "debate is martingale")
>   - Takata, Masumori & Ikegami → arXiv:2509.04537 (não "Iwasaki")
>   - Dochkina (2026) → preprint MIPT, autor único, citar com cautela
>   - MapCoder-Lite → arXiv:2509.17489
>   - Cemri et al. (MAST) → arXiv:2503.13657
> - Faltam datas exatas e venues para várias citações (Brown 2020, Bonabeau 1999, etc.)
> - Considerar adicionar Phi-4 (não só Phi-3) — mais recente, mais relevante
> - Seção 2.5 (benchmarks) é longa — pode ser comprimida; foco é justificar HumanEval, não revisar todos
> - Falta seção sobre **PID (Partial Information Decomposition)** — só mencionada em passagem. Se for usar conceito no cap 6 (trabalhos futuros), expandir aqui

## 2.1 Modelos de Linguagem Pequenos (SLMs)

O paradigma dominante em inteligência artificial generativa tem sido a escalabilidade: modelos maiores, treinados em mais dados, produzem resultados melhores (Kaplan et al., 2020). Entretanto, a partir de 2023, uma linha de pesquisa emergente demonstrou que modelos pequenos (<10B parâmetros), quando treinados com dados de alta qualidade, podem igualar ou superar modelos muito maiores em tarefas específicas.

### 2.1.1 Evidências de Eficiência

O relatório técnico do Phi-3 (Abdin et al., 2024) demonstrou que um modelo de 3.8B parâmetros pode igualar o desempenho do Mixtral 8×7B (56B parâmetros efetivos) em benchmarks de raciocínio, atribuindo o resultado à qualidade superior dos dados de treinamento sintéticos. Esta descoberta desafiou a suposição de que escala é pré-requisito para capacidade.

Trabalhos subsequentes confirmaram o padrão:

| Modelo | Parâmetros | Resultado |
|--------|------------|-----------|
| Phi-4 (Abdin et al., 2024) | 14B | Supera GPT-4 em STEM |
| Qwen2.5-Coder (Qwen Team, 2024) | 7B | 84.1% HumanEval (vs. 67% GPT-4 original) |
| TinyLLaMA (Zhang et al., 2024) | 1.1B | Desempenho comparável a modelos 3× maiores |

### 2.1.2 Técnicas de Compressão

A viabilidade de SLMs em hardware de consumo depende de técnicas de quantização que reduzem requisitos de memória sem degradação significativa de desempenho:

- **GPTQ** (Frantar et al., 2023): Quantização pós-treinamento para 3-4 bits, ICLR 2023
- **AWQ** (Lin et al., 2024): Quantização ciente de ativação, Best Paper MLSys 2024
- **QLoRA** (Dettmers et al., 2023): Fine-tuning eficiente em 4-bit, NeurIPS 2023

Estas técnicas permitem executar modelos de 7B parâmetros em GPUs com 8GB de VRAM, democratizando o acesso a capacidades avançadas de geração de código.

### 2.1.3 SLMs para Tarefas Agênticas

Belcak e Heinrich (2025), em artigo de posição da NVIDIA Research, argumentam que SLMs são suficientes para a maioria das tarefas agênticas, propondo um algoritmo de conversão LLM→SLM para destilação de capacidades. O argumento central é que tarefas agênticas tipicamente requerem execução de ações atômicas bem-definidas, onde a capacidade de raciocínio complexo de LLMs maiores é subutilizada.

Esta perspectiva motiva a presente pesquisa: se SLMs individuais são capazes, populações de SLMs coordenados podem amplificar esta capacidade?

---

## 2.2 Sistemas Multi-Agente com LLMs

A aplicação de LLMs em arquiteturas multi-agente representa uma convergência entre duas tradições: sistemas multi-agente clássicos da inteligência artificial distribuída e as capacidades emergentes de modelos de linguagem modernos.

### 2.2.1 Frameworks Estabelecidos

Os principais frameworks multi-agente para LLMs surgiram entre 2023-2024:

**MetaGPT** (Hong et al., 2024) introduziu Procedimentos Operacionais Padrão (SOPs) para estruturar a colaboração entre agentes com papéis explícitos (Gerente de Produto, Arquiteto, Engenheiro, QA). O sistema alcançou 85.9% no HumanEval, demonstrando que coordenação estruturada amplifica capacidade individual. Publicado como Oral no ICLR 2024 (top 1.2%).

**AutoGen** (Wu et al., 2024) propôs o paradigma de "programação por conversação", onde desenvolvedores definem agentes e padrões de interação, e a execução emerge do diálogo entre agentes. Publicado no COLM 2024.

**ChatDev** (Qian et al., 2024) modelou o desenvolvimento de software como uma "empresa virtual" com agentes assumindo papéis organizacionais. O sistema gera software funcional em menos de 7 minutos em média. Publicado no ACL 2024.

**CAMEL** (Li et al., 2023) introduziu o paradigma de role-playing com "inception prompting", onde agentes recebem instruções sobre seus papéis e objetivos. Publicado no NeurIPS 2023.

### 2.2.2 Padrões de Orquestração

Guo et al. (2024), em survey apresentado no IJCAI, identificaram quatro padrões de orquestração:

| Padrão | Descrição | Exemplo |
|--------|-----------|---------|
| Centralizado | Supervisor coordena agentes especializados | MetaGPT |
| Descentralizado | Agentes negociam diretamente | CAMEL |
| Hierárquico | Planejador decompõe; executores implementam | HuggingGPT |
| Debate | Agentes argumentam até consenso | Multi-Agent Debate |

### 2.2.3 Debate Multi-Agente

Du et al. (2024) propuseram que múltiplos LLMs debatendo podem superar agentes individuais em tarefas de raciocínio, publicado no ICML 2024. Entretanto, trabalhos recentes questionam esta premissa:

Zhang et al. (2025) em "Stop Overvaluing Multi-Agent Debate" demonstraram que grande parte dos ganhos atribuídos ao debate podem ser replicados por técnicas mais simples como self-consistency.

Choi et al. (arXiv:2508.17536, NeurIPS 2025 Spotlight) demonstraram formalmente que **debate entre LLMs é um martingale** — a precisão esperada não melhora com rodadas adicionais de debate. A votação majoritária captura a maior parte dos ganhos observados, questionando se a complexidade adicional se justifica.

### 2.2.4 Limitações dos Sistemas Atuais

La Malfa et al. (2025) em "Large Language Models Miss the Multi-Agent Mark" apresentaram crítica metodológica contundente: a maioria das reivindicações de emergência em sistemas multi-agente com LLMs são metodologicamente fracas, arriscando "inflar argumentos sobre inteligência geral de LLMs".

Os autores identificam três problemas recorrentes:

1. **Ausência de baselines apropriados**: Comparações com agente único sem controle de compute budget
2. **Métricas inadequadas**: Foco em desempenho final sem caracterizar mecanismos
3. **Reivindicações inflacionadas**: Uso impreciso de "emergência" e "inteligência coletiva"

Cemri et al. (arXiv:2503.13657, NeurIPS 2025 Datasets & Benchmarks) complementam esta crítica com uma taxonomia sistemática de modos de falha em sistemas multi-agente. O framework MAST identifica 14 modos de falha distintos, documentando taxas de falha entre 41% e 86.7% em benchmarks estabelecidos. Notavelmente, modelos da família Qwen2.5 foram incluídos na avaliação, fornecendo baseline direto para a presente pesquisa.

Esta crítica motiva a presente pesquisa a empregar métricas operacionalizadas, baselines controlados, e cautela nas reivindicações.

---

## 2.3 Emergência e Auto-Organização

O conceito de emergência — propriedades do sistema que não podem ser previstas a partir das propriedades dos componentes individuais — é central para esta pesquisa. Distinguimos emergência genuína de diferenciação trivial ou ruído estatístico.

### 2.3.1 Fundamentos Teóricos

A inteligência de enxame (swarm intelligence) emergiu do estudo de sistemas biológicos descentralizados: colônias de formigas, enxames de abelhas, cardumes de peixes. Bonabeau et al. (1999) formalizaram os princípios:

1. **Interações locais**: Agentes respondem a vizinhos imediatos, não a estado global
2. **Feedback positivo**: Comportamentos bem-sucedidos são amplificados
3. **Feedback negativo**: Saturação previne colapso em comportamento único
4. **Flutuações**: Aleatoriedade permite exploração de novas soluções

Casadei et al. (2023), em survey publicado na Artificial Life, mapearam a engenharia de inteligência coletiva artificial, identificando tensões entre controle top-down e emergência bottom-up.

### 2.3.2 Emergência em Sistemas de LLM

A aplicação do framework de swarm intelligence a LLMs é recente e contestada. Rahman e Schranz (2025) em "LLM-Powered Swarms: A New Frontier or a Conceptual Stretch?" questionam se a metáfora é produtiva:

> "Enquanto sistemas biológicos de enxame emergem de agentes simples com regras fixas, LLMs são agentes complexos com comportamento estocástico. A analogia pode obscurecer mais do que ilumina."

Jimenez-Romero et al. (2025), por outro lado, demonstraram comportamento emergente em simulações de enxame controladas por LLMs, onde padrões de coordenação surgiram sem programação explícita.

### 2.3.3 Aprendizado In-Context como Mecanismo

Brown et al. (2020) estabeleceram que LLMs podem "aprender" durante inferência através de exemplos no contexto (in-context learning), sem atualização de pesos. Este mecanismo é central para a presente pesquisa:

- Agentes acumulam exemplos bem-sucedidos no contexto
- Contextos diferentes produzem comportamentos diferentes
- Diferenciação emerge sem treinamento explícito

Agarwal et al. (2024), no NeurIPS, demonstraram que ICL escala com número de exemplos e pode sobrepor vieses do pré-treinamento — evidência de que contexto acumulado pode genuinamente modificar comportamento.

### 2.3.4 Diferenciação em Populações Idênticas

Takata, Masumori e Ikegami (arXiv:2509.04537, ALIFE 2025) investigaram o problema de El Farol Bar com agentes LLM idênticos, demonstrando que:

> "Cada agente não é controlado por um motor LLM separado, mas pelo mesmo motor subjacente, com individualidade emergindo apenas de diferenças em memória e histórico de interação."

Este resultado é precedente direto para a presente pesquisa: agentes idênticos podem diferenciar-se através de experiência acumulada, sem diferenças arquiteturais.

---

## 2.4 O Limiar de Capacidade

Trabalhos recentes estabeleceram uma descoberta crítica: existe um limiar de capacidade abaixo do qual auto-organização em populações de LLM falha. Esta seção revisa as duas contribuições seminais que motivam o reframing da presente pesquisa.

### 2.4.1 O Framework de Riedl (2025)

Riedl (arXiv:2510.05174, ICLR 2026 poster) investigou emergência de coordenação em populações de LLMs idênticos através de um framework de quatro testes:

1. **Critério prático**: O sistema resolve a tarefa?
2. **Capacidade de emergência**: Informação mútua entre agentes aumenta? (via PID)
3. **Teste de coalizão**: Subgrupos de agentes preservam funcionalidade?
4. **Diferenciação de agentes**: Agentes desenvolvem comportamentos distintos?

O resultado central para a presente pesquisa é a **condição "Plain"**: agentes idênticos congelados, sem prompts de persona, com feedback escalar de grupo. Nesta condição:

- **Llama-3.1-8B falhou** com apenas 10% de sucesso
- Diferenciação observada foi **drift estocástico**, não especialização estável
- Persona prompts foram necessários para induzir papéis funcionais

**Implicação**: Modelos abaixo de ~8B parâmetros podem ser incapazes de auto-organização em configurações padrão.

### 2.4.2 A Hipótese de Dochkina (2026)

Dochkina (arXiv:2603.28990) propõe, em preprint ainda não revisado por pares, que abaixo de um limiar de capacidade, auto-organização pode prejudicar desempenho. A autora reporta experimentos com 25.000 tarefas e 8 modelos, comparando auto-organização versus atribuição rígida de papéis.

**Nota metodológica**: Este trabalho deve ser citado com cautela. Trata-se de preprint de autor único (MIPT), que referencia modelos não existentes em abril de 2026 ("GPT-5.4", "Gemini-3-flash"). Os effect sizes específicos reportados (Cohen's d = 1.86, +14%, +44%) requerem validação independente.

Embora os números específicos aguardem confirmação, a **hipótese qualitativa** — de que auto-organização pode prejudicar modelos menores — é consistente com achados independentes de Riedl (2025) e Cemri et al. (2025).

A explicação proposta: modelos menores não possuem capacidade suficiente para simultaneamente (a) resolver a tarefa e (b) coordenar com outros agentes. A carga cognitiva da coordenação consome capacidade que deveria ser alocada à tarefa.

### 2.4.3 Evidência Adicional: MapCoder-Lite

MapCoder-Lite (arXiv:2509.17489) fornece evidência direta do limiar no contexto específico de geração de código multi-agente. O trabalho demonstra que modelos de 7B parâmetros colapsam em configurações multi-agente sem intervenção específica:

- **Vanilla 7B**: 13.2% de sucesso em coordenação multi-agente
- **Com destilação direcionada**: 28.3% de sucesso

Este resultado sugere que o limiar de capacidade não é absoluto, mas permeável com técnicas de post-training apropriadas.

> 📝 **NOTA SEÇÃO 2.4:** Esta é a seção MAIS importante do capítulo — estabelece o problema central (limiar). Verificar se a transição entre Riedl/Dochkina/MapCoder está fluida. Após ter resultados de sweep, talvez vale citar trabalhos adicionais sobre o limiar específico de 7B.

### 2.4.4 Implicações para a Presente Pesquisa

Os achados de Riedl, Dochkina e MapCoder-Lite estabelecem o problema central desta dissertação:

1. **O limiar existe**: Auto-organização falha abaixo de certa capacidade
2. **A localização é incerta**: Riedl indica ~8B; Dochkina indica >10B
3. **O mecanismo de feedback importa**: Ambos usaram feedback sintético ou LLM-judge

A questão que emerge: **feedback verificável (testes unitários) pode quebrar o limiar?**

Diferentemente de feedback escalar (Riedl) ou julgamento por LLM (Dochkina), testes unitários fornecem sinal binário, objetivo, e não-ambíguo. Esta diferença pode reduzir a carga cognitiva de interpretação do feedback, liberando capacidade para especialização.

---

## 2.5 Benchmarks de Código

A avaliação de capacidade de geração de código em LLMs evoluiu rapidamente, com preocupações crescentes sobre contaminação de dados e saturação de benchmarks estabelecidos.

### 2.5.1 HumanEval e MBPP

**HumanEval** (Chen et al., 2021) consiste em 164 problemas de programação Python, cada um com docstring descritiva e testes unitários. A métrica padrão é Pass@k — a probabilidade de pelo menos uma solução correta em k tentativas.

**MBPP** (Austin et al., 2021) oferece 500 problemas de nível introdutório, complementando HumanEval com maior volume estatístico.

Ambos os benchmarks enfrentam limitações em 2026:

| Problema | Evidência |
|----------|-----------|
| Saturação | Modelos frontier excedem 95% Pass@1 |
| Contaminação | Yang et al. (2023) documentaram ~25% overlap com dados de treino do GPT-4 |
| Cobertura limitada | Testes originais podem não capturar edge cases |

### 2.5.2 EvalPlus e Variantes Robustas

Liu et al. (2023), no NeurIPS, propuseram **EvalPlus** — extensão do HumanEval e MBPP com 80× mais testes unitários por problema. O benchmark expôs que soluções "corretas" frequentemente falhavam em casos não cobertos pelos testes originais:

- GPT-4: 88% HumanEval → 79% HumanEval+
- CodeLlama-34B: 53% HumanEval → 45% HumanEval+

Para a presente pesquisa, utilizamos categorização de tarefas do HumanEval original, mas reconhecemos que avaliação futura deveria empregar HumanEval+ para maior rigor.

### 2.5.3 SWE-bench

**SWE-bench** (Jimenez et al., 2024, ICLR Oral) elevou o padrão de avaliação ao usar issues reais do GitHub. O dataset original contém 2.294 issues de 12 repositórios Python populares (Django, Flask, Scikit-learn, etc.).

Variantes do benchmark:

| Variante | Tamanho | Status (2026) |
|----------|---------|---------------|
| SWE-bench Full | 2.294 | Computacionalmente proibitivo |
| SWE-bench Lite | 300 | Superseded |
| SWE-bench Verified | 500 | Recomendado atual |

**Nota importante**: SWE-bench Lite foi oficialmente substituído por SWE-bench Verified em 2025, após análise revelar testes defeituosos em 59.4% das instâncias difíceis não resolvidas.

### 2.5.4 Benchmarks de Nova Geração

Em resposta às limitações de HumanEval e contaminação crescente, surgiram benchmarks resistentes:

**LiveCodeBench** (2025) utiliza problemas de LeetCode e Codeforces publicados após a data de corte dos modelos, garantindo ausência de contaminação por construção. O benchmark é atualizado continuamente.

**BigCodeBench** (2025) avalia uso realista de bibliotecas (numpy, pandas, requests), testando não apenas sintaxe mas integração com ecossistema Python.

### 2.5.5 Escolha de Benchmark para Esta Pesquisa

Selecionamos HumanEval como benchmark principal pelos seguintes critérios:

| Critério | HumanEval | SWE-bench |
|----------|-----------|-----------|
| Feedback verificável | Sim (testes unitários) | Sim |
| Tempo de execução | Segundos | Minutos-horas |
| Variância preservada para SLMs | Sim (40-80% range) | Baixa (<5%) |
| Diversidade de tipos | 4 categorias claras | Heterogêneo |

Para modelos de 1.5B-7B parâmetros, HumanEval preserva variância suficiente (Pass@1 entre 40-85%) para detectar diferenças entre condições experimentais. SWE-bench, embora mais realista, apresentaria floor effects (desempenho próximo a 0%) que impediriam análise de especialização.

Reconhecemos esta limitação: generalização dos achados para tarefas de engenharia de software do mundo real requer validação futura em benchmarks como SWE-bench Verified ou Terminal-Bench.

---

> 📝 **NOTA:** Seção 2.6 (Contradições) foi adicionada por sugestão da análise mas pode parecer "ajuste de contas" com a literatura — característica não-tradicional em ABNT. Avaliar se manter como seção dedicada ou diluir nas seções respectivas (2.1, 2.2, 2.3). Argumento PRO manter: força crítica explícita; CONTRA: quebra fluxo narrativo.

## 2.6 Contradições na Literatura

A literatura sobre sistemas multi-agente com SLMs apresenta contradições importantes que esta pesquisa deve engajar explicitamente.

### 2.6.1 Otimismo vs. Falhas Empíricas

Belcak e Heinrich (2025) argumentam que SLMs são suficientes para tarefas agênticas, citando reduções de custo de 10-30×. Entretanto, estudos empíricos documentam taxas de falha de 41-86.7% (Cemri et al., 2025) e overhead de coordenação de até 300× (Rahman & Schranz, 2025).

**Reconciliação proposta:** A eficácia de SLMs depende do tipo de tarefa. Single-agent function calling com schema constraints é viável; multi-agent coordination com contexto compartilhado não é. A presente pesquisa testa esta distinção empiricamente.

### 2.6.2 Inversão de Rewards em Pequena Escala

A narrativa popular sugere que verifiable rewards (RLVR) fecham a lacuna de modelos pequenos. Evidência recente (arXiv:2604.02621) inverte esta expectativa: em modelos de 125M-350M, judge rewards superam verifiable rewards por 5-10 pontos em raciocínio matemático. Em 6.7B, a diferença é insignificante (~1.5 pontos).

**Implicação:** Modelos pequenos precisam de densidade de sinal, não necessariamente correção de sinal. Verifiable rewards sozinhos podem ser insuficientes.

### 2.6.3 Multi-Agent Debate como Martingale

AgentGroupChat-V2 reporta +11pp em MATH-L5 com multi-agent, enquanto Choi et al. (NeurIPS 2025) provam que debate é martingale — a precisão esperada não melhora com rodadas adicionais.

**Reconciliação:** AgentGroupChat usa task decomposition + heterogeneous backbones, não debate puro. Ganhos vêm de majority voting + heterogeneidade de modelos, não de refinamento iterativo.

---

## 2.7 Síntese e Lacunas

A revisão da literatura revela uma lacuna específica que a presente pesquisa endereça:

| Trabalho Existente | Contribuição | Lacuna |
|--------------------|--------------|--------|
| MetaGPT, ChatDev | Papéis explícitos via prompts | Não investiga emergência |
| Riedl (2025) | Framework para medir emergência | Usa feedback escalar sintético |
| Dochkina (2026) | Propõe limiar de capacidade | Preprint não revisado |
| NVIDIA SLM (2025) | Argumenta viabilidade de SLMs | Não valida empiricamente |
| Takata et al. (2025) | Diferenciação por memória | Domínio não-código |
| Cemri et al. (2025) | Taxonomia de modos de falha | Não testa SLMs explicitamente |

**A lacuna**: Nenhum trabalho investiga se feedback verificável (testes unitários) permite emergência de especialização em SLMs abaixo do limiar documentado por Riedl e proposto por Dochkina.

A presente pesquisa preenche esta lacuna através de:

1. Experimentos com modelos de 1.5B, 3B e 7B parâmetros
2. Feedback binário objetivo (testes passam/falham)
3. Métricas operacionalizadas de especialização
4. Design experimental que permite resultado negativo publicável
