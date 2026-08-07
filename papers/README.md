# Artigos Científicos

**Status: PENDENTE — Fase 13 (Publicação) não foi iniciada.**

Nenhum manuscrito foi escrito. Este diretório não contém um artigo
fabricado — a Fase 2 (`docs/science/pergunta-cientifica.md`) definiu a
pergunta de pesquisa, hipóteses e venues-alvo, mas um artigo real só faz
sentido depois que:

1. A Fase 6 (Machine Learning) produzir resultados comparáveis contra o
   baseline (Fase 5) e contra a literatura (Fase 2, seção "Lacunas").
2. A Fase 12 (Validação Científica) rodar — benchmark contra baseline,
   validação temporal, estudos de ablação, quantificação de incerteza.

## Estrutura planejada quando a Fase 13 começar

```
papers/
├── draft-v1/
│   ├── manuscript.md          # estrutura IMRaD
│   ├── figures/
│   └── references.bib
└── submission-log.md          # onde foi submetido, status, feedback de revisão
```

## Onde estão as decisões que vão alimentar o artigo, desde já

- `docs/science/pergunta-cientifica.md` — problema, hipóteses (H1–H3), contribuição científica posicionada, lacunas na literatura com citações.
- `docs/science/REFERENCES.md` — referências levantadas via busca ativa (não são só recordação de treinamento — foram verificadas contra a literatura corrente de 2025/2026).
- `CHANGELOG.md` — histórico de decisões e bugs corrigidos, material bruto para a seção de metodologia/limitações.
