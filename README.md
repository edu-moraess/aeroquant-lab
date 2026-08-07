# AeroQuant Lab

Plataforma de pesquisa para monitoramento inteligente da saúde de
aeronaves — Digital Twin, predição de RUL, manutenção preditiva.

**Status: Fase 5 de 13 concluída.** Núcleo científico priorizado: RUL
Prediction + Digital Twin com quantificação de incerteza (decisão
registrada em `docs/architecture/fase1-auditoria-arquitetura.md`, seção 1.5).
Este README descreve o que **existe de verdade** — para o que está
planejado mas não implementado, ver `docs/roadmap.md`.

## Status por fase

| Fase | Nome | Status |
|---|---|---|
| 1 | Auditoria e Arquitetura | Completo |
| 2 | Pergunta Científica | Completo |
| 3 | Engenharia de Dados | Completo, testado |
| 4 (Nível 1) | Dados Sintéticos | Completo, testado |
| 4 (Níveis 2-3) | Dados Públicos / Treino Híbrido | Bloqueado — aguardando upload do C-MAPSS (`data/external/`) |
| 5 | Digital Twin | Completo, testado (baseline estatístico) |
| 6 | Machine Learning | Não iniciado |
| 7 | Computer Vision | Não iniciado (bloqueio científico: sem dataset real licenciado) |
| 8 | Simulação Monte Carlo | Não iniciado |
| 9 | Explainable AI | Não iniciado |
| 10 | Dashboard | Esqueleto escrito, não testado (sem rede no dev original) |
| 11 | Engenharia/MLOps | CI/Docker existem; MLflow/DVC não integrados |
| 12 | Validação Científica | Não iniciado |
| 13 | Publicação | Não iniciado |

## Instalação rápida

```bash
git clone <este-repositório>
cd AeroQuant_Lab
bash scripts/install.sh
source .venv/bin/activate
```

Ou manualmente:

```bash
pip install -r requirements/base.txt
PYTHONPATH=src python3 -m unittest discover tests/unit -v
PYTHONPATH=src python3 -m unittest discover tests/integration -v
```

## Rodando as demos

```bash
PYTHONPATH=src python3 scripts/demo_end_to_end.py    # Sensor Data + ETL
PYTHONPATH=src python3 scripts/demo_digital_twin.py   # Digital Twin + RUL
```

Ou abra `notebooks/01_pipeline_overview.ipynb` (testado — roda ponta a
ponta sem erro, ver `docs/developer/DEVELOPER_GUIDE.md`).

## Estrutura do repositório

```
AeroQuant_Lab/
├── src/aeroquant/              # código-fonte (Clean Architecture por Bounded Context)
│   ├── sensor_data/             # Fase 3-4: gerador sintético, ETL, qualidade, adaptador C-MAPSS
│   ├── digital_twin/            # Fase 5: Health Index, RUL+incerteza, anomalia
│   └── api/                     # Fase 10 (parcial): FastAPI — NAO TESTADO
├── tests/
│   ├── unit/                    # 9 testes
│   └── integration/             # 4 testes
├── scripts/                     # demos executáveis + instalação
├── notebooks/                   # notebook testado (roda ponta a ponta)
├── dashboards/                  # Streamlit — NAO TESTADO
├── docs/
│   ├── architecture/             # decisões de arquitetura, Fases 1 e 5
│   ├── science/                  # pergunta científica, hipóteses, referências
│   ├── developer/                 # guia de desenvolvimento
│   ├── researcher/                # guia de uso científico
│   ├── api/                       # documentação da API
│   ├── roadmap.md
│   └── backlog.md
├── data/
│   ├── sample/                   # dataset sintético de exemplo (gerado, incluído)
│   └── external/                  # C-MAPSS — vazio, aguardando upload (ver README lá dentro)
├── models/                       # vazio de propósito — Fase 6 não implementada (ver README lá dentro)
├── papers/                       # vazio de propósito — Fase 13 não iniciada (ver README lá dentro)
├── assets/diagrams/               # diagramas Mermaid (arquitetura + sequência do Digital Twin)
├── config/                        # YAML — parâmetros do gerador e do Digital Twin
├── docker/                        # Dockerfile + docker-compose.yml
├── .github/workflows/ci.yml       # CI real (roda a suíte de testes)
├── requirements/                   # separado por grupo (base/dev/api/dashboard/mlops/ml)
├── pyproject.toml
├── CHANGELOG.md                    # inclui os 3 bugs reais encontrados e corrigidos
├── CONTRIBUTING.md
└── LICENSE                         # MIT (default assumido, ver nota no próprio arquivo)
```

## Três bugs reais encontrados durante o desenvolvimento

Detalhes completos em `CHANGELOG.md` e `docs/architecture/fase5-digital-twin.md`:

1. Sensores saturando o teto do range válido aos ~15% da vida útil (escala errada do efeito de degradação).
2. Baseline de Health Index contaminado pela própria degradação da unidade monitorada (Health Index caindo perto do fim de vida, o oposto do esperado).
3. Limiar de falha arbitrário (`1.0`) sem relação com a escala real do Health Index, corrigido com calibração empírica.

Isso não é uma lista de vergonha — é o tipo de rastro que uma auditoria
científica real deveria deixar. Ver `docs/developer/DEVELOPER_GUIDE.md`
para o padrão comum entre os três (constantes assumidas sem validação
empírica) e como evitar repetir.

## O que este projeto NÃO finge ter

Por decisão explícita (ver aviso em cada README correspondente):
- `models/` não tem modelo treinado (Fase 6 não existe ainda).
- `papers/` não tem manuscrito (Fase 13 não começou).
- `src/aeroquant/api/` e `dashboards/` têm código real mas **não testado**
  neste ambiente (sem acesso à rede durante o desenvolvimento).
- `data/external/` não tem os arquivos C-MAPSS (bloqueado por falta de
  acesso à internet no ambiente de desenvolvimento original — precisam
  ser enviados manualmente).

## Documentação

- Arquitetura completa: `docs/architecture/fase1-auditoria-arquitetura.md`
- Pergunta científica e hipóteses: `docs/science/fase2-pergunta-cientifica.md`
- Decisões específicas do Digital Twin: `docs/architecture/fase5-digital-twin.md`
- Roadmap: `docs/roadmap.md` | Backlog: `docs/backlog.md`
- Para desenvolvedores: `docs/developer/DEVELOPER_GUIDE.md`
- Para pesquisadores: `docs/researcher/RESEARCHER_GUIDE.md`

## Licença

MIT — ver `LICENSE` (inclui nota sobre por que essa foi a escolha padrão).
