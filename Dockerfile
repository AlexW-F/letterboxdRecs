# letterboxdRecs API — Python 3.11 + Surprise + implicit + sentence-transformers-ready
#
# Build:   docker build -t letterboxd-recs .
# Run:     docker run -p 8000:8000 -v $(pwd)/models:/app/models \
#                                  -v $(pwd)/ml-32m:/app/ml-32m \
#                                  -v $(pwd)/data:/app/data \
#                                  letterboxd-recs
#
# scikit-surprise needs to compile against the *installed* numpy, so the
# build is two-stage to avoid baking gcc into the runtime image.

FROM python:3.11-slim AS build
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Build deps for surprise (Cython + gcc) and implicit (numpy headers).
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc g++ \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY requirements.txt ./
# Pin numpy<2 because scikit-surprise wheels are built against numpy 1.x.
RUN pip install --no-cache-dir 'numpy<2' 'pandas<3' 'scipy<1.14' \
 && pip install --no-cache-dir -r requirements.txt \
 && pip install --no-cache-dir implicit diskcache pyarrow

FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OPENBLAS_NUM_THREADS=1
WORKDIR /app
COPY --from=build /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=build /usr/local/bin/uvicorn /usr/local/bin/uvicorn
COPY src /app/src
COPY scripts /app/scripts

# Models, datasets, and content features should be mounted at runtime.
ENV MODELS_DIR=/app/models \
    ML_DATA_DIR=/app/ml-32m \
    CACHE_DIR=/app/.api_cache \
    CONTENT_FEATURES=/app/data/content_features \
    SVD_FILE=svd_full.pkl \
    ALS_FILE=als_full.pkl

EXPOSE 8000
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
