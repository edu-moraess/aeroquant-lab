# AeroQuant Lab — Fase 2: Pergunta Científica

*Escopo fixado na Fase 1: núcleo científico = RUL Prediction + Digital Twin com quantificação de incerteza. As demais fases (CV, simulação, dashboard) permanecem como módulos de suporte ao redor desse núcleo.*

## 1. Problema de Pesquisa

A predição de RUL (Remaining Useful Life) em motores turbofan é dominada por abordagens de *Domain Adaptation* (DA), que transferem conhecimento de domínios-fonte ricos em dados para domínios-alvo escassos, mitigando os desvios de distribuição causados por diferentes condições operacionais — uma revisão recente de 2025 sobre DA profunda para RUL de turbofans caracteriza exatamente esse cenário como o principal motivador do campo.

O que a literatura majoritariamente faz é transferir entre **subconjuntos reais** do próprio benchmark (ex.: FD001 → FD002 dentro do C-MAPSS). O problema de pesquisa do AeroQuant Lab é mais específico: **quanto de um gerador sintético *controlável* de degradação (domain randomization) pode substituir dados reais escassos, e como esse gap synthetic-to-real pode ser medido e reduzido sistematicamente via fine-tuning, dentro de uma arquitetura de Digital Twin que expõe incerteza como cidadã de primeira classe (não como pós-processamento)?**

## 2. Hipóteses

- **H1 (transferência híbrida)**: um modelo pré-treinado em dados sintéticos e ajustado (fine-tuned) com uma fração pequena de dados públicos reais supera, em regimes de poucos dados-alvo, um modelo treinado apenas nos dados públicos disponíveis — padrão já reportado em estudos de transfer learning aplicados a sinais industriais (não específico deste projeto; ainda não testado aqui, depende dos dados C-MAPSS reais).

- **H2 (quantificação do gap)**: o desempenho relativo entre "modelo treinado só em sintético" e "modelo fine-tuned" pode ser expresso como função mensurável dos parâmetros do gerador estocástico (nível de ruído, taxa de deriva, variância de degradação) — permitindo responder "quanto realismo sintético é necessário" em vez de tratar isso como escolha arbitrária.

- **H3 (incerteza como decisão)**: um Health Index com incerteza quantificada (aleatória + epistêmica) produz decisões de manutenção mensuravelmente mais conservadoras e seguras do que uma estimativa pontual de RUL — indo na mesma direção de frameworks recentes que aprendem incerteza aleatória diretamente via modelagem probabilística em cima do C-MAPSS, descritos como abordagem ainda pouco explorada nessa literatura específica.

## 3. Objetivos

**Objetivo geral**: desenvolver e validar uma metodologia de treinamento híbrido sintético→real para RUL, integrada a um Digital Twin com quantificação de incerteza, usando motores turbofan (C-MAPSS/N-CMAPSS) como caso de estudo.

**Objetivos específicos**:
1. Projetar um gerador sintético de sensores validável estatisticamente (propriedades de ruído, deriva e degradação testadas via `hypothesis`/testes de propriedade).
2. Quantificar o gap sintético-real como função dos parâmetros do gerador.
3. Desenvolver um modelo de RUL com incerteza aleatória aprendida + incerteza epistêmica via ensemble/MC-Dropout.
4. Integrar o modelo a um motor de Digital Twin que atualiza Health Index e RUL incrementalmente a partir de um stream simulado.
5. Comparar contra baselines estabelecidos (LSTM, Bi-LSTM com atenção, CNN-autoencoder) nos mesmos subconjuntos do C-MAPSS/N-CMAPSS, respeitando a função de penalidade assimétrica da NASA (penaliza mais superestimar RUL do que subestimar), como usada em trabalhos recentes de referência.

## 4. Contribuição Científica

Decidi **não** posicionar a contribuição como "mais uma arquitetura de RUL" — esse espaço já está saturado por variações de atenção, Bi-LSTM e autoencoders convolucionais, como evidenciam múltiplos trabalhos de 2025 usando essencialmente a mesma base (C-MAPSS) com módulos de atenção incrementais. A contribuição do AeroQuant Lab é **metodológica**, em duas frentes:

- **(a) Tratar a geração sintética como objeto de estudo empírico**, não como etapa incidental de augmentation — caracterizando sistematicamente o gap sintético-real em função dos parâmetros do gerador (H2), algo que as revisões de domain adaptation para turbofans apontam como direção de pesquisa ainda pouco padronizada.
- **(b) Arquitetura de referência reprodutível** que integra DA híbrida + Digital Twin + incerteza + explicabilidade em um único pipeline versionado (MLflow/DVC), servindo como base comparável para trabalhos futuros — a maioria dos trabalhos revisados publica resultados de modelo isolado, sem arquitetura de sistema completa acompanhando o artigo.

**Alternativas de posicionamento consideradas e rejeitadas**:
- *Foco em nova arquitetura de rede neural*: rejeitado — baixo potencial de diferenciação dado o volume de publicações recentes com pequenas variações de atenção/LSTM sobre C-MAPSS.
- *Foco exclusivo em Computer Vision de inspeção*: rejeitado nesta fase — como já registrado na auditoria da Fase 1, não há dataset público relevante e licenciado; a contribuição ficaria cientificamente frágil sem dados reais.

## 5. Lacunas na Literatura

1. **Transferência a partir de gerador sintético controlável, não apenas entre subconjuntos reais**: a revisão de domain adaptation de 2025 foca em técnicas de DA entre domínios de dados reais existentes, identificando desafios e tendências futuras, mas sem tratar a caracterização paramétrica de geradores sintéticos como direção central.
2. **Incerteza desacoplada da estratégia de transferência**: quantificação de incerteza em RUL é um tema ativo, mas normalmente tratado como característica isolada do modelo, raramente combinada explicitamente com uma estratégia de domain adaptation sintético→real.
3. **Digital Twins aeroespaciais concentrados em maquinário rotativo/estrutural, com pouca integração de ponta a ponta**: uma revisão de estado da arte de 2025 sobre métodos inspirados em Digital Twin cobre principalmente diagnóstico de falhas e RUL em maquinário rotativo, sem uma arquitetura de referência aberta e reprodutível que uma cadeia completa de dados sintéticos → DT → explicabilidade.

## 6. Potencial de Publicação

**Venues identificados como ativos agora (2026)**:
- **PHM Society Conference (North America) 2026** — chamada de artigos aberta, com Data Challenge em paralelo (edição atual focada em estimativa de dano em dentes de engrenagem, mas a trilha geral de artigos aceita RUL/prognóstico como tema central).
- **PHM Europe 2026**, em Oslo, 8–10 de julho de 2026 — evento presencial confirmado da PHM Society.
- Precedente direto de aderência do tema: a própria PHM Society já organizou um desafio de dados dedicado a motores turbofan em 2021, confirmando que o tópico é bem recebido pela comunidade.
- **Journals de segunda etapa** (após validação mais madura): *Reliability Engineering & System Safety* e *Mechanical Systems and Signal Processing* aparecem repetidamente como veículos de referência nos trabalhos revisados nesta fase.
- **Caminho realista**: publicar primeiro como *preprint* (arXiv) + submissão a um Data Challenge ou trilha de artigo da PHM Society NA/EU 2026 — não almejar diretamente periódico de alto impacto no primeiro ciclo; isso é consistente com o volume observado de contribuições PHM Society vindas de projetos de mestrado/pesquisa aplicada, não apenas grandes laboratórios.

## 7. Limitações Assumidas Desde Já

- O gerador sintético não terá validação física (não é simulação de elementos finitos) — é estocástico, calibrado estatisticamente contra o comportamento agregado do C-MAPSS/N-CMAPSS, não contra física de degradação real de turbinas.
- "Tempo real" do Digital Twin é replay simulado de séries temporais, não hardware físico.
- A contribuição (a) depende de conseguir de fato caracterizar uma relação sistemática gap↔parâmetros — se o resultado for ruidoso/sem padrão claro, isso também é um resultado científico válido (resultado negativo documentado), mas muda o ângulo do artigo.

## 8. Status (atualizado — este documento hoje vive em `docs/science/`)

Checkpoint original (histórico): antes da Fase 3, ficou em aberto se o
projeto usaria N-CMAPSS ou C-MAPSS clássico. **Decisão tomada**: C-MAPSS
clássico (FD001–FD004) — é o benchmark mais usado nos trabalhos
comparáveis levantados acima, o que facilita comparação direta de
resultados futuros contra a literatura. Ver `docs/roadmap.md` para o
status atual de todas as fases e `CHANGELOG.md` para o histórico de
decisões desde então.
