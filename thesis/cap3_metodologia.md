# 3. METODOLOGIA

> 📝 **NOTAS GERAIS PARA REVISÃO DESTE CAPÍTULO:**
> - Capítulo escrito ANTES da implementação completa estar finalizada — verificar consistência com src/ atual.
> - **Limitação importante a destacar**: desenho single-seed por configuração. Mencionar explicitamente em alguma seção (3.X de "Limitações Metodológicas Conhecidas"?) ou só no cap 5.
> - Verificar se as métricas S, D, F descritas aqui correspondem exatamente ao que é calculado em src/swarm/metrics.py.
> - Falta discutir hardware utilizado (GPU, RAM, sistema operacional) — exigido por ABNT em alguns programas.
> - Falta mencionar que o teste de razão de verossimilhança (LR test) foi adicionado como análise principal de diferenciação funcional, complementando o qui-quadrado original.
> - A figura ASCII pode ser substituída por figura proper (TikZ ou Mermaid → PDF) na versão final.

## 3.1 Visão Geral

Este capítulo descreve a arquitetura do sistema multi-agente, as métricas de emergência empregadas, e o design experimental utilizado para investigar o limiar de capacidade para especialização emergente.

A metodologia segue três princípios:

1. **Aprendizado apenas por contexto**: Todos os modelos permanecem congelados (sem atualização de pesos). A diferenciação entre agentes ocorre exclusivamente pela acumulação de exemplos no contexto.

2. **Feedback verificável**: O sinal de sucesso/falha provém da execução de testes unitários, não de avaliação por LLM ou heurísticas.

3. **Reprodutibilidade**: Todos os experimentos utilizam sementes fixas, configurações declarativas em YAML, e modelos de código aberto.

---

## 3.2 Arquitetura do Sistema

O sistema implementa uma população de N agentes idênticos coordenados por um roteador central. A Figura 1 ilustra a arquitetura.

```
┌─────────────────────────────────────────────────────┐
│                     ROTEADOR                        │
│  ┌─────────────────────────────────────────────┐   │
│  │ Affinity Router (softmax sobre histórico)   │   │
│  └─────────────────────────────────────────────┘   │
│         │              │              │            │
│         ▼              ▼              ▼            │
│   ┌──────────┐   ┌──────────┐   ┌──────────┐      │
│   │ Agente 0 │   │ Agente 1 │   │ Agente 2 │      │
│   │ Context  │   │ Context  │   │ Context  │      │
│   │ Buffer   │   │ Buffer   │   │ Buffer   │      │
│   └──────────┘   └──────────┘   └──────────┘      │
│         │              │              │            │
│         └──────────────┼──────────────┘            │
│                        ▼                           │
│              ┌─────────────────┐                   │
│              │  Modelo Base    │                   │
│              │  (Qwen2.5-Coder)│                   │
│              │   Congelado     │                   │
│              └─────────────────┘                   │
└─────────────────────────────────────────────────────┘
```

### 3.2.1 Agentes

Cada agente mantém um estado composto por:

- `context_buffer`: Lista de até K exemplos bem-sucedidos (problema, solução)
- `task_history`: Histórico de sucesso por tipo de tarefa
- `affinity_scores`: Taxas de sucesso acumuladas por categoria

Todos os agentes compartilham o mesmo modelo base. A individualidade emerge apenas das diferenças no `context_buffer` — exemplos distintos produzem prompts distintos, que produzem comportamentos distintos.

### 3.2.2 Roteador

O roteador seleciona qual agente processará cada tarefa. Implementamos dois tipos:

**Random Router** (controle): Seleção uniforme aleatória.

**Affinity Router** (experimental): Seleção proporcional ao histórico de sucesso do agente para o tipo de tarefa, usando softmax com temperatura τ:

```
P(agente_i | tipo_t) = exp(score_i,t / τ) / Σ_j exp(score_j,t / τ)
```

Onde `score_i,t` é a taxa de sucesso do agente i em tarefas do tipo t. Temperatura alta (τ → ∞) aproxima distribuição uniforme; temperatura baixa (τ → 0) concentra em agentes de melhor desempenho.

---

> 📝 **NOTA SEÇÃO 3.3:** Verificar com cuidado as fórmulas. As métricas devem ser apresentadas como ADAPTAÇÕES de medidas estabelecidas, não invenções. Em particular:
> - S é equivalente normalizada da informação mútua (coeficiente de incerteza de Theil 1970)
> - D é relacionada à divergência de Vendi (Vendi Score)
> - F deve mencionar tanto o teste qui-quadrado tradicional QUANTO o teste de razão de verossimilhança (LR) que foi adicionado como análise mais rigorosa
> Após reescrita, garantir consistência total entre as definições aqui e o uso no cap 4.

## 3.3 Métricas de Emergência

Para quantificar a emergência de especialização, adaptamos três métricas estabelecidas na literatura de teoria da informação, ecologia quantitativa, e aprendizado multi-agente.

### 3.3.1 Índice de Especialização (S)

O Índice de Especialização quantifica o grau em que agentes específicos tornam-se associados a tipos específicos de tarefa. Adaptamos o coeficiente de incerteza de Theil (1970), equivalente à informação mútua normalizada, amplamente utilizado em ecologia para medir especialização de nicho (Blüthgen et al., 2006) e em aprendizado por reforço multi-agente como objetivo de diversidade (Wang et al., 2020 - ROMA).

A métrica é definida como:

```
S = 1 - H(tipo | agente) / H(tipo)
```

Onde:
- H(tipo) é a entropia da distribuição de tipos de tarefa
- H(tipo | agente) é a entropia condicional dado o agente atribuído

**Interpretação:**
- S = 0: Atribuição aleatória; saber o agente não informa o tipo de tarefa
- S = 1: Especialização perfeita; cada agente processa apenas um tipo
- S ∈ (0, 1): Especialização parcial

A significância estatística é avaliada por teste de permutação (n = 1000), comparando S observado contra a distribuição nula gerada por embaralhamento das atribuições agente-tarefa.

### 3.3.2 Divergência de Contexto (D)

A Divergência de Contexto mede quão diferentes são os exemplos acumulados por cada agente. Esta métrica é análoga ao Vendi Score (Friedman & Dieng, 2022) e ao Specialization Efficiency Index (SEI) proposto em trabalhos recentes de sistemas multi-agente (arXiv:2510.07888).

```
D = 1 - média(sim_cos(embed(C_i), embed(C_j))) para todos os pares (i,j)
```

Onde:
- C_i é a concatenação dos exemplos no contexto do agente i
- embed() é uma função de embedding (all-MiniLM-L6-v2)
- sim_cos é a similaridade de cosseno

**Interpretação:**
- D = 0: Contextos idênticos entre agentes
- D = 1: Contextos completamente ortogonais
- D crescente ao longo do tempo indica diferenciação

### 3.3.3 Diferenciação Funcional (F)

A Diferenciação Funcional avalia se agentes apresentam desempenho estatisticamente diferente entre tipos de tarefa — isto é, se a especialização tem consequência funcional.

Empregamos um modelo linear generalizado de efeitos mistos (GLMM):

```
sucesso ~ tipo_tarefa * agente + (1|problema)
```

O termo de interação `tipo_tarefa * agente` captura se o efeito do tipo de tarefa varia entre agentes. A significância deste termo indica diferenciação funcional.

Esta abordagem segue a recomendação de Riedl (2025), que propõe um framework de quatro testes para validar emergência em sistemas multi-agente. Nossa implementação representa uma versão simplificada — um "snapshot estático" — que não captura a dinâmica temporal nem separa redundância de sinergia via decomposição de informação parcial (PID). Reconhecemos esta limitação e indicamos a análise temporal como trabalho futuro.

### 3.3.4 Relação entre as Métricas

As três métricas capturam aspectos complementares da emergência:

| Métrica | O que mede | Limitação |
|---------|------------|-----------|
| S | Associação agente-tarefa | Não distingue causalidade |
| D | Diversidade de experiência | Necessária mas não suficiente |
| F | Consequência funcional | Não captura dinâmica temporal |

Uma reivindicação robusta de emergência requer S > 0 significativo, D > 0.3, e F significativo simultaneamente.

---

## 3.4 Design Experimental

### 3.4.1 Benchmark

Utilizamos o HumanEval (Chen et al., 2021) como benchmark principal. O dataset consiste em 164 problemas de programação em Python, cada um especificando uma função a ser implementada com docstring descritiva e testes unitários para validação.

Escolhemos HumanEval pelos seguintes critérios:

1. **Feedback verificável**: Cada solução é validada por execução de testes, não por avaliação subjetiva
2. **Diversidade de tarefas**: Problemas abrangem manipulação de strings, matemática, estruturas de dados, e lógica
3. **Escala apropriada**: 164 problemas permitem múltiplas iterações sem saturação estatística
4. **Comparabilidade**: Benchmark amplamente reportado na literatura de modelos de código

**Categorização de Tarefas**

Classificamos os 164 problemas em quatro categorias para análise de especialização:

| Categoria | Descrição | Quantidade |
|-----------|-----------|------------|
| STRING | Manipulação de texto, parsing, formatação | 41 |
| MATH | Operações numéricas, algoritmos matemáticos | 76 |
| LIST | Estruturas de dados, arrays, transformações | 37 |
| LOGIC | Condicionais, validação, lógica booleana | 10 |

A categorização foi realizada manualmente com base na operação predominante de cada problema.

### 3.4.2 Modelos

Todos os experimentos utilizam modelos da família Qwen2.5-Coder (Qwen Team, 2024), selecionados por:

1. **Licença permissiva**: Apache 2.0 (1.5B) e Qwen Research License (3B, 7B)
2. **Otimização para código**: Pré-treinamento específico em corpora de programação
3. **Disponibilidade de tamanhos**: Família completa de 0.5B a 32B parâmetros
4. **Compatibilidade com quantização**: Formato GGUF para inferência eficiente

| Modelo | Parâmetros | Quantização | VRAM | Arquivo |
|--------|------------|-------------|------|---------|
| Qwen2.5-Coder-1.5B-Instruct | 1.5B | Q4_K_M | ~2GB | 1.0 GB |
| Qwen2.5-Coder-3B-Instruct | 3B | Q4_K_M | ~3GB | 1.9 GB |
| Qwen2.5-Coder-7B-Instruct | 7B | Q4_K_M | ~5GB | 4.4 GB |

### 3.4.3 Condições Experimentais

O design experimental varia três fatores independentes:

| Fator | Níveis | Justificativa |
|-------|--------|---------------|
| Tamanho do modelo | 1.5B, 3B, 7B | Mapear limiar de capacidade |
| Temperatura do roteador | 0.1, 0.3, 0.5 | Exploração vs. explotação |
| Tamanho da população | 3, 5 agentes | Pressão competitiva |

A Tabela 4 descreve as condições principais:

| Experimento | Modelo | Agentes | Temp | Objetivo |
|-------------|--------|---------|------|----------|
| exp_baseline | 1.5B | 3 | 0.5 | Linha de base |
| exp_low_temp | 1.5B | 3 | 0.1 | Efeito da temperatura |
| exp_5_agents | 1.5B | 5 | 0.3 | Efeito da população |
| exp_3b_baseline | 3B | 3 | 0.5 | Linha de base 3B |
| exp_3b_model | 3B | 3 | 0.3 | Ponto intermediário |
| exp_3b_low_temp | 3B | 3 | 0.1 | Temperatura no 3B |
| exp_3b_5_agents | 3B | 5 | 0.3 | População no 3B |
| exp_7b_model | 7B | 3 | 0.3 | Capacidade alta |

Adicionalmente, foram conduzidos sweeps sistemáticos de temperatura (oito valores: 0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5) e população (oito valores: n = 3, 4, 5, 6, 7, 8, 9, 10) em todos os três tamanhos de modelo, totalizando experimentos adicionais para caracterização robusta dos efeitos.

> 📝 **NOTA:** Tabela atualizada com todas condições principais. Os SWEEPS estão sendo executados — quando concluídos, talvez vale uma sub-tabela separada listando-os, ou apenas referência cruzada para a seção de resultados.

### 3.4.4 Protocolo de Execução

Cada experimento segue o protocolo:

1. Inicializar N agentes com contextos vazios
2. Para cada problema p em HumanEval (ordem fixa):
   a. Classificar tipo de tarefa
   b. Roteador seleciona agente
   c. Construir prompt com exemplos do contexto do agente
   d. Gerar solução
   e. Executar testes unitários
   f. Se sucesso: adicionar (problema, solução) ao contexto do agente
   g. Atualizar histórico de afinidade
3. Computar métricas finais (S, D, F)

O tamanho máximo do contexto é K = 5 exemplos por agente, gerenciado por política FIFO (first-in, first-out).

---

## 3.5 Configuração de Hardware e Software

### 3.5.1 Hardware

Todos os experimentos foram executados em uma única estação de trabalho:

- **GPU**: NVIDIA RTX 4070 Laptop (8GB VRAM)
- **CPU**: Intel Core i7 (12 núcleos)
- **RAM**: 32GB DDR5
- **Armazenamento**: SSD NVMe 1TB

Esta configuração representa hardware de consumo acessível, alinhada com o objetivo de investigar a "fronteira de acessibilidade" para sistemas multi-agente locais.

### 3.5.2 Software

| Componente | Versão | Função |
|------------|--------|--------|
| Python | 3.11 | Linguagem principal |
| llama-cpp-python | 0.2.x | Inferência de modelos GGUF |
| sentence-transformers | 2.x | Embeddings para divergência |
| statsmodels | 0.14 | Modelos de efeitos mistos |
| scipy | 1.11 | Testes estatísticos |

### 3.5.3 Reprodutibilidade

Para garantir reprodutibilidade:

- Semente aleatória fixa (42) para todas as operações estocásticas
- Configurações declarativas em arquivos YAML
- Código-fonte disponível em repositório público
- Modelos obtidos de Hugging Face com hashes verificados
