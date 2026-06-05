# 6. CONCLUSÃO

> 📝 **NOTAS GERAIS PARA REVISÃO DESTE CAPÍTULO:**
> - Capítulo curto (intencionalmente) — Conclusão deve ser síntese, não repetição extensa do cap 5.
> - Verificar redundância com 4.5 (Síntese) e 5.6 (Síntese). Se sobreposição grande, eliminar uma das três sínteses.
> - "Considerações Finais" (6.4) tem tom mais filosófico/reflexivo — apropriado para Conclusão de mestrado em PT-BR. Verificar se programa permite/incentiva.
> - **Trabalhos Futuros (6.3)** é longo. Pode ser visto como wishlist de boas ideias (positivo) OU como dispersão (negativo). Considerar trimar para 4-5 itens centrais.
> - Falta menção explícita à pendência de validação por múltiplas sementes. Isso é uma limitação que define agenda futura — pode ser destacada.
> - A frase "epifenômenos estatísticos" (último parágrafo) é elegante mas pesada. Verificar se traduz a intenção corretamente.

## 6.1 Resposta às Questões de Pesquisa

A presente dissertação investigou a emergência de especialização funcional em populações de Modelos de Linguagem Pequenos (SLMs) idênticos, utilizando feedback verificável de testes unitários. Os experimentos conduzidos em modelos de 1.5B, 3B e 7B parâmetros (Qwen2.5-Coder), com variações de temperatura de roteamento e tamanho de população, permitem responder às três questões de pesquisa formuladas.

### 6.1.1 RQ1: Existência do Limiar de Capacidade

**Existe um limiar de capacidade para emergência de especialização em populações de SLMs com feedback verificável.**

Na configuração de linha de base (temperatura 0.5, três agentes), tanto o modelo 1.5B (S = 0.009, p = 0.767) quanto o modelo 3B (S = 0.022, p = 0.221) falharam em produzir especialização significativa. Apenas o modelo 7B (S = 0.116, p < 0.001) apresentou emergência espontânea, com diferenciação funcional genuína confirmada pelo teste de razão de verossimilhança (χ² = 15.60, p = 0.008).

A localização do limiar — entre 3B e 7B parâmetros para emergência espontânea — é consistente com o achado de Riedl (2025) de que Llama-3.1-8B já apresenta limitações de auto-organização. A presente pesquisa estende este achado ao demonstrar que **feedback verificável binário não é suficiente** para quebrar o limiar em modelos sub-limiar, refutando parcialmente a hipótese inicial de que ambiguidade de feedback fosse o fator limitante.

### 6.1.2 RQ2: Mecanismos Moduladores

**A localização efetiva do limiar é modulada por temperatura de roteamento e tamanho de população, mas com efeitos qualitativamente diferentes em cada escala de modelo.**

Identificamos três regimes distintos:

- **Sub-limiar (1.5B)**: Intervenções induzem especialização, mas com trade-off negativo de desempenho (-3.6 pontos percentuais em Pass@1 com baixa temperatura). A "especialização" observada é primariamente concentração de roteamento, sem diferenciação funcional substantiva.

- **Limiar (3B)**: Intervenções induzem especialização **sem** trade-off de desempenho. O experimento 3B com baixa temperatura produziu o maior índice de especialização observado (S = 0.390) com Pass@1 mantido em 81.1%. Os agentes desenvolveram perfis funcionais distintos em diferentes tipos de tarefa.

- **Supra-limiar (7B)**: Emergência espontânea sem necessidade de intervenção, com diferenciação funcional genuína detectada por testes estatísticos rigorosos.

> 📝 **NOTA:** A resposta a RQ3 (a seguir) é a mais delicada. Há tensão entre:
> - "feedback verificável é necessário mas não suficiente" (interpretação cautelosa, alinhada com evidência)
> - "feedback verificável quebra parcialmente o limiar" (interpretação otimista, defensável mas mais fraca)
> Decidir tom. Atual versão é cautelosa. Se quiser argumentar contribuição mais forte, reescrever.

### 6.1.3 RQ3: Suficiência do Feedback Verificável

**Feedback verificável é necessário, mas não suficiente, para emergência em modelos abaixo do limiar.**

Os resultados refutam parcialmente a hipótese de que a ambiguidade de feedback (escalar sintético em Riedl, LLM-judge em Dochkina) era o fator principal limitando emergência em modelos pequenos. Mesmo com sinal binário objetivo de testes unitários, o modelo 1.5B na configuração padrão não desenvolve especialização, e o modelo 3B requer intervenções no roteador. A capacidade base do modelo permanece como pré-requisito.

Entretanto, feedback verificável é claramente **necessário**: sua ausência (em sistemas que dependem de auto-avaliação ou debate) introduz fontes adicionais de ruído que provavelmente exacerbam as limitações de capacidade. A presente pesquisa estabelece um piso: o que é alcançável com o sinal mais limpo possível.

## 6.2 Contribuições

A presente dissertação oferece quatro contribuições principais:

1. **Mapeamento empírico do limiar de capacidade** em uma família coerente de modelos (Qwen2.5-Coder 1.5B/3B/7B). O isolamento do efeito de escala — mantendo arquitetura, dados de treinamento e quantização constantes — fortalece a validade interna da comparação.

2. **Refinamento conceitual do limiar como gradiente, não fronteira**. A identificação de três zonas (sub-limiar, limiar, supra-limiar) com efeitos qualitativamente diferentes de intervenções fornece um framework mais informativo que o modelo binário implícito na literatura.

3. **Identificação do desaparecimento do trade-off especialização-desempenho** em modelos próximos ao limiar. Esta descoberta tem valor prático imediato: identifica configurações específicas (3B parâmetros, baixa temperatura) que produzem sistemas multi-agente eficazes em hardware de consumo.

4. **Distinção metodológica entre concentração de roteamento e diferenciação funcional**. A aplicação do teste de razão de verossimilhança sobre regressão logística com interação revela que padrões reportados como "especialização" frequentemente refletem assimetrias de carga, não diferenças substantivas em capacidade. Esta distinção alinha-se com as preocupações metodológicas de La Malfa et al. (2025).

## 6.3 Trabalhos Futuros

Os achados desta pesquisa abrem várias direções de investigação:

### 6.3.1 Replicação e Generalização

**Múltiplas sementes**: Replicar todos os experimentos com 5-10 sementes diferentes para obter intervalos de confiança robustos sobre os efeitos identificados.

**Outros benchmarks**: Estender a investigação para SWE-bench Verified (Jimenez et al., 2024) e Terminal-Bench, benchmarks que aproximam-se mais de tarefas de engenharia de software realistas. A natureza heterogênea destes benchmarks pode revelar padrões de especialização não capturados em HumanEval.

**Outras famílias de modelos**: Replicar com Phi-3, Llama-3.2, e Gemma-2 para verificar se a localização do limiar é específica do Qwen ou generalizável.

### 6.3.2 Extensões Metodológicas

**Designs ensemble**: Permitir múltiplos agentes a tentarem o mesmo problema, com agregação por votação. Isto habilitaria análise GLMM completa com efeito aleatório por problema, alinhando-se metodologicamente com Riedl (2025).

**Sweeps mais finos**: Os experimentos atuais variam temperatura em 0.1, 0.3, 0.5 e população em 3, 5. Sweeps mais finos (e.g., temperatura 0.05-1.5 em incrementos de 0.1) podem identificar pontos de transição precisos.

**Métricas baseadas em PID**: Adotar a decomposição parcial de informação (Riedl, 2025) para distinguir sinergia de redundância em populações multi-agente, fornecendo medida mais granular de emergência.

### 6.3.3 Investigações Teóricas

**Feedback denso**: A inversão observada por Choi et al. (2024) — onde modelos pequenos beneficiam-se mais de feedback denso (LLM-judge) que de verifiable rewards — sugere que o tipo de sinal pode interagir com escala. Experimentos comparando feedback binário com feedback graduado (e.g., código parcialmente correto) podem revelar mecanismos cognitivos subjacentes.

**Coordenação síncrona vs. assíncrona**: O design atual emprega roteamento sequencial. Coordenação simultânea (debate, voting) introduce dinâmicas distintas e pode apresentar limiares diferentes.

**Modelos pós-treinados para coordenação**: O resultado de MapCoder-Lite (arXiv:2509.17489) sugere que post-training específico pode quebrar o limiar mesmo em modelos 7B. Investigar se SLMs especificamente fine-tuned para coordenação (não apenas para a tarefa) podem operar abaixo do limiar identificado.

### 6.3.4 Aplicações

**Sistemas de produção**: Implementar sistemas reais que aproveitem as configurações identificadas (3B, baixa temperatura) para tarefas práticas como assistência de código, análise de bugs e refatoração automatizada.

**Hardware educacional**: Desenvolver pacotes de software que permitam a estudantes e pesquisadores com hardware modesto experimentarem com sistemas multi-agente especializados, democratizando o acesso à pesquisa em sistemas de IA distribuídos.

## 6.4 Considerações Finais

Esta pesquisa partiu de uma questão técnica — sobre limiares de capacidade em SLMs — mas alcança implicações mais amplas sobre a natureza da emergência em sistemas computacionais. Os achados sugerem que emergência genuína não é uma propriedade que aparece subitamente em uma escala, mas se desenvolve gradualmente, com a possibilidade de ser induzida em escalas menores pelo design cuidadoso do sistema de coordenação.

A identificação de zonas qualitativamente distintas — onde a mesma intervenção produz resultados opostos (especialização degenerada vs. saudável) — desafia hipóteses simplistas sobre a relação entre escala e capacidade. Sistemas multi-agente eficazes em SLMs são possíveis, mas requerem alinhamento cuidadoso entre capacidade base do modelo e mecanismos de coordenação.

Para o campo mais amplo, esta pesquisa reforça uma lição metodológica: medidas de "emergência" devem ser robustas a artefatos de design experimental. A distinção entre concentração de roteamento e diferenciação funcional substantiva — invisível a testes qui-quadrado padrão — exemplifica como reivindicações de inteligência coletiva podem ser inflacionadas por escolhas analíticas inadequadas. Em uma era de progresso rápido na inteligência artificial, esta cautela metodológica é essencial para distinguir avanços genuínos de epifenômenos estatísticos.
