#!/usr/bin/env bash
# Instalação do núcleo testado. Para extras (api/dashboard/mlops/ml),
# ver requirements/ — não testados no ambiente original de desenvolvimento.
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"

echo "== Verificando versão do Python =="
"$PYTHON_BIN" --version

echo "== Criando ambiente virtual (.venv) =="
"$PYTHON_BIN" -m venv .venv
source .venv/bin/activate

echo "== Instalando dependências do núcleo =="
pip install --upgrade pip
pip install -r requirements/base.txt

echo "== Rodando testes (unit + integration) =="
PYTHONPATH=src python -m unittest discover tests/unit -v
PYTHONPATH=src python -m unittest discover tests/integration -v

echo "== OK. Rode 'source .venv/bin/activate' e depois:"
echo "   PYTHONPATH=src python3 scripts/demo_end_to_end.py"
echo "   PYTHONPATH=src python3 scripts/demo_digital_twin.py"
