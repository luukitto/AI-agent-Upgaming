# Docker image for Render (free web service). Data lives in Qdrant Cloud; the container is stateless.
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    BM25_CACHE_PATH=/tmp/bm25.pkl PORT=10000

# run as an unprivileged user
RUN useradd -m -u 1000 user
WORKDIR /home/user/app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY --chown=user app app
COPY --chown=user scripts scripts
COPY --chown=user eval eval
COPY --chown=user data/synthetic data/synthetic
RUN mkdir -p data && chown user data

USER user
# bake the embedding model into the image so cold starts don't download it
RUN python -c "from app.embeddings import LocalEmbedder; LocalEmbedder('BAAI/bge-small-en-v1.5', 'data/models')"
EXPOSE 10000
CMD ["sh", "-c", "uvicorn app.api.main:app --host 0.0.0.0 --port ${PORT}"]
