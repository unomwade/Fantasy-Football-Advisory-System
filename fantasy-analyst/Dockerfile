FROM python:3.12-slim

WORKDIR /app

# build-essential: duckdb/chromadb wheels occasionally need build tools
# curl: required by the HEALTHCHECK below
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# data/ is overridden by the docker-compose volume mount at runtime;
# this just ensures the directory exists if someone runs the image standalone.
RUN mkdir -p /app/data/chroma

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
