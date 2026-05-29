FROM python:3.12-slim

WORKDIR /app

# Dipendenze di sistema
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ && \
    rm -rf /var/lib/apt/lists/*

# Dipendenze Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Codice e modelli
COPY tennis_api.py tennis_dataset.py tennis_models.py tennis_config.py tennis_analysis.py ./
COPY models/ ./models/

# HF Spaces usa porta 7860
ENV PORT=7860
EXPOSE 7860

CMD ["gunicorn", "tennis_api:app", "--bind", "0.0.0.0:7860", "--workers", "1", "--timeout", "120"]
