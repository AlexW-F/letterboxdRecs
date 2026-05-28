# Hugging Face Spaces (Docker SDK) image for the movienight API.
#
# Unlike the repo-root Dockerfile (which mounts the heavy blobs at runtime for
# local docker-compose), this one BAKES the trained artifacts into the image —
# they're committed to the Space repo via git-LFS. Build context = Space root.
#
# Expected layout at the Space repo root (see deploy/hf/DEPLOY notes):
#   Dockerfile  README.md  requirements-api.txt  .gitattributes
#   src/  scripts/
#   models/  -> svd_full_slim.pkl, als_full.pkl
#   data/    -> content_genome.npz, content_genome.json, popularity.json,
#               (optional) movie_space_index.pkl
#   ml-32m/  -> movies.csv, links.csv   (ratings.csv NOT needed — popularity.json replaces it)

FROM python:3.11-slim AS build
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc g++ \
 && rm -rf /var/lib/apt/lists/*
WORKDIR /build
COPY requirements-api.txt ./requirements.txt
# numpy<2 first: scikit-surprise wheels are built against numpy 1.x.
RUN pip install --no-cache-dir 'numpy<2' 'pandas<3' \
 && pip install --no-cache-dir -r requirements.txt

FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OPENBLAS_NUM_THREADS=1
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
 && rm -rf /var/lib/apt/lists/*
COPY --from=build /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=build /usr/local/bin/uvicorn /usr/local/bin/uvicorn

# HF Spaces runs the container as uid 1000.
RUN useradd -m -u 1000 user
WORKDIR /app

COPY --chown=user src /app/src
COPY --chown=user scripts /app/scripts
COPY --chown=user models /app/models
COPY --chown=user data /app/data
COPY --chown=user ml-32m /app/ml-32m

ENV MODELS_DIR=/app/models \
    ML_DATA_DIR=/app/ml-32m \
    CONTENT_FEATURES=/app/data/content_genome \
    POPULARITY_FILE=/app/data/popularity.json \
    MOVIE_SPACE_INDEX=/app/data/movie_space_index.pkl \
    CACHE_DIR=/tmp/api_cache \
    SVD_FILE=svd_full_slim.pkl \
    ALS_FILE=als_full.pkl
# Set these in the HF Space settings (Variables/Secrets), NOT here:
#   CORS_ALLOW_ORIGINS=https://movienight.alex-wf.com
#   TMDB_API_KEY=<your key>   (optional; only for server-side CSV enrichment)

USER user
EXPOSE 7860
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "7860"]
