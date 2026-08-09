# AeroQuant Lab — Relatório de status

**Objetivo do projeto:** pesquisa em monitoramento inteligente da saúde de motores turbofan (C-MAPSS-like), com Digital Twin, estimativa de RUL com incerteza, ML/redes neurais, detecção de anomalias e explicabilidade — com honestidade científica sobre dados sintéticos.

**App:** https://aeroquant-lab.streamlit.app/  
**Repositório:** https://github.com/edu-moraess/aeroquant-lab

---

## 1. Visão geral do que funciona

O núcleo operacional está implementado de ponta a ponta: **gerar dados → estimar saúde → prever vida restante → quantificar incerteza → comparar modelos → explicar → detectar anomalias → visualizar no dashboard**.

Arquitetura em *bounded contexts* (DDD leve):

| Context | Papel |
|---------|--------|
| `sensor_data` | Schema C-MAPSS-like, gerador estocástico, ETL |
| `digital_twin` | HI, RUL OLS, baseline Welford, anomalia local |
| `ml` | Trainers, split por unidade, métricas NASA |
| `uncertainty` | Monte Carlo (aleatória / epistêmica) |
| `xai` | SHAP (TreeSHAP quando aplicável) |
| `anomaly` | Isolation Forest + residual z-score |
| `api` | Endpoints FastAPI (espelho do núcleo) |

Dashboard unificado (Streamlit), **sem multipage**, tema **nativo** (Light/Dark via ⋮ → Settings).

---

## 2. Por que cada peça faz sentido

### 2.1 Dados sintéticos (`sensor_data`)
- **O que faz:** gera trajetórias run-to-failure com degradação gradual/abrupta, ruído e acoplamento por sensor.
- **Por que:** C-MAPSS real ainda não está ingestado; o gerador permite desenvolver e validar *pipelines* sem vazamento de dados e sem bloquear o restante da pesquisa.
- **Limite honesto:** resultados numéricos **não** são generalizáveis para frota real até haver dados públicos/reais.

### 2.2 Digital Twin (`digital_twin`)
- **HI (Health Index):** z-score ponderado por acoplamento de degradação + baseline online (Welford) por condição operacional.
- **RUL:** extrapolação linear do HI até limiar **calibrado** na frota sintética, com intervalo de predição OLS e nível de confiança configurável (80/90/95%).
- **Anomalia embutida:** salto de HI com z-score local.
- **Por que:** é o baseline estatístico clássico em prognósticos — interpretável, barato e referência obrigatória antes de ML mais complexo (exigência metodológica do projeto).

### 2.3 Machine Learning clássico
- Modelos: **Linear**, **Random Forest**, **GBM quantile**, **MLP**.
- **Split por `unit_id`:** evita vazamento temporal (linhas da mesma unidade são correlacionadas).
- Métricas: RMSE, MAE e **NASA asymmetric score** (penaliza superestimar RUL — erro operacionalmente perigoso).
- **Por que:** comparar algoritmos antes da escolha final; o NASA score alinha a avaliação ao PHM Challenge / literatura C-MAPSS.

### 2.4 Redes neurais e sequência
- **MLP tabular:** features engenheiradas (roll, delta, z-score).
- **Sequence MLP:** janelas `(T × F)` achatadas + MLP — baseline sequencial **sem torch**.
- **LSTM:** opcional se `torch` estiver instalado.
- **Por que:** degradação é temporal; janelas capturam dependência no tempo. Sequence MLP garante que o Cloud continue funcional sem dependência pesada.

### 2.5 Monte Carlo (`uncertainty`)
- Decomposição **empírica** de variância do RUL em componente aleatória (ruído/seed) e epistêmica (incerteza do limiar).
- **Por que:** sustenta a hipótese H3 de incerteza de forma operacional, sem fingir ser Bayesiana formal.

### 2.6 XAI (`xai`)
- TreeSHAP em RF/GBM: importância global e explicação local.
- **Por que:** em manutenção preditiva, a predição precisa ser auditável; SHAP explica o **modelo**, não a física do motor (limitação documentada na UI).

### 2.7 Anomalias (`anomaly`)
- **Isolation Forest:** treino no início de vida (regime “saudável”), score na vida inteira.
- **Residual z-score:** alinhado à lógica do Digital Twin.
- **Por que:** falhas abruptas e saltos de sensores são sinais de alerta distintos do RUL contínuo; dualidade detector estatístico + ensemble é padrão em PHM.

### 2.8 UI/UX
- Uma app, abas claras, metodologias em expanders (o quê / como / interpretar / limitações).
- Modebar Plotly discreto; gráficos com `theme="streamlit"`.
- **Por que:** apresentação científica e demo profissional sem poluir a sidebar nem forçar tema custom que quebra Light/Dark.

---

## 3. O que cada aba do dashboard entrega

| Aba | Entrega |
|-----|---------|
| **Digital Twin** | RUL vs verdadeiro, HI, residual, incerteza OLS, sensores |
| **Fleet** | Heatmap HI, ranking, distribuição por modo de falha, trajetórias médias |
| **ML clássico** | Comparação Linear/RF/GBM/MLP, residual, SHAP |
| **Neural Net** | MLP tabular · Sequence MLP · LSTM (se torch) |
| **Anomalias** | IF ou z-score, ranking, timeline score+HI, acordo com DT |
| **Monte Carlo** | Distribuição de RUL e variâncias aleatória/epistêmica |

---

## 4. Testes automatizados existentes

- Unit: gerador/ETL, métricas ML e split, Monte Carlo, SHAP.
- Integration: Digital Twin, comparação ML.

(Cobertura ainda não é “validação científica formal” — isso permanece no roadmap.)

---

## 5. O que **não** está pronto (e por quê)

| Item | Motivo |
|------|--------|
| C-MAPSS real | Depende de download/licença/dados externos |
| Treino híbrido sintético→real | Depende do item acima |
| Transformer RUL | Próximo no roadmap; custo/complexidade > Sequence MLP no Cloud |
| Computer Vision | Fora do núcleo RUL atual |
| MLOps completo (MLflow/DVC) | Prioridade menor até haver dados reais |
| Incerteza Bayesiana formal | Monte Carlo é empírica por desenho |

---

## 6. Coerência científica (resumo)

1. **Baseline antes de complexidade** — DT estatístico → ML clássico → MLP → sequência → LSTM.  
2. **Sem vazamento** — split por unidade.  
3. **Métrica alinhada ao domínio** — NASA score.  
4. **Incerteza explícita** — IC OLS, quantis GBM, Monte Carlo, contamination IF.  
5. **Limitações declaradas na UI** — dados sintéticos, SHAP ≠ física, LSTM opcional.

Isso torna o lab **defendável em apresentação técnica**: não promete o que os dados não sustentam, mas demonstra o pipeline completo de prognóstico.

---

## 7. Próximos passos recomendados

1. Transformer leve (torch opcional) para RUL sequencial.  
2. Loader C-MAPSS (FD001) + mesmo schema/ETL.  
3. Validação formal: hipóteses H1–H3, tabelas de benchmark, testes de robustez.  
4. Opcional: `torch` no `requirements.txt` do Cloud **só** se o deploy aguentar o tamanho.

---

*Documento gerado a partir do estado do repositório (dashboard unificado + contexts listados acima).*
