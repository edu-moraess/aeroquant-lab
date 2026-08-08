# AeroQuant Lab

Plataforma de pesquisa para monitoramento inteligente da saúde de
aeronaves — Digital Twin, predição de RUL, manutenção preditiva.

**Status: Fase 6 de 13 concluída.** Núcleo científico priorizado: RUL
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
| 6 | Machine Learning | Completo — 3 modelos comparados contra o baseline (ver CHANGELOG) |
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