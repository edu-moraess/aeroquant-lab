# Contribuindo com o AeroQuant Lab

## Princípios de engenharia (não negociáveis, herdados do master prompt do projeto)

1. **Nunca superficial.** Toda decisão técnica precisa de justificativa e comparação com alternativas — ver exemplos em `docs/architecture/` e nos comentários de módulo em `src/`.
2. **Clean Architecture estrita.** `domain/` nunca importa de `infrastructure/` ou `application/`. Trocar uma implementação (ex.: CSV → Postgres) deve ser uma troca de `infrastructure`, nunca exigir mudança em `domain`/`application`.
3. **Nenhum resultado sem teste.** Todo módulo novo em `src/aeroquant/<context>/` precisa de testes correspondentes em `tests/unit/` ou `tests/integration/`.
4. **Bugs encontrados vão para o `CHANGELOG.md`**, com a causa raiz explicada — não só "corrigido X", mas por que X estava errado.

## Estrutura para adicionar um novo Bounded Context

```
src/aeroquant/<novo_context>/
├── domain/            # entidades e value objects, sem dependências externas
├── application/       # ports (Protocol) + use cases
└── infrastructure/     # implementações concretas das ports
```

## Rodando os testes

```bash
pip install -r requirements/dev.txt
PYTHONPATH=src python3 -m unittest discover tests/unit -v
PYTHONPATH=src python3 -m unittest discover tests/integration -v
# ou, com pytest instalado:
pytest tests/ -v
```

## Ambiente sem acesso à rede (como este foi desenvolvido)

Partes deste projeto foram escritas em um ambiente sem acesso à internet —
por isso `pydantic`, `pyarrow`, `DVC`, `MLflow`, `FastAPI`, `Streamlit`,
`pytest` e `hypothesis` aparecem em `requirements/*.txt` mas não foram
testados diretamente contra o código que os usa (`src/aeroquant/api/`,
`dashboards/`). Esses arquivos estão claramente marcados com um aviso no
topo — rode os testes desses módulos específicos assim que tiver rede
disponível, antes de assumir que estão corretos.

## Antes de abrir um PR (quando este projeto virar um repositório de verdade)

- [ ] Testes novos cobrindo o comportamento adicionado
- [ ] `CHANGELOG.md` atualizado
- [ ] Se mudou uma decisão de arquitetura: atualizar `docs/architecture/`
- [ ] Se mudou uma escolha de dado/modelo relevante à pergunta científica: atualizar `docs/science/`
