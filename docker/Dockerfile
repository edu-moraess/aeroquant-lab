# Imagem mínima para o núcleo testado (Sensor Data + Digital Twin Context).
# Extras (api, dashboard, mlops, ml) não testados neste ambiente — ver
# CONTRIBUTING.md. Adicionar seus requirements*.txt ao build quando forem
# validados.
FROM python:3.12-slim

WORKDIR /app

COPY requirements/base.txt requirements/base.txt
RUN pip install --no-cache-dir -r requirements/base.txt

COPY pyproject.toml .
COPY src/ src/
COPY tests/ tests/
COPY scripts/ scripts/
COPY config/ config/

ENV PYTHONPATH=/app/src

# Sanidade: roda a suíte de testes no build, falha o build se algo quebrar.
RUN python -m unittest discover tests/unit -v && \
    python -m unittest discover tests/integration -v

CMD ["python", "scripts/demo_digital_twin.py"]
