# DepthWizard — standalone web container (FastAPI + Three.js frontend).
#
# CPU inference by default (works anywhere). For the NVIDIA GPU image rebuild
# with a CUDA index:
#   docker build --build-arg TORCH_INDEX=cu124 -t depthwizard:gpu .

FROM python:3.11-slim

ARG TORCH_INDEX=cpu

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DW_DEVICE=auto

WORKDIR /app

# Minimal system libs needed by numpy/scipy/rasterio/matplotlib wheels.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        libglib2.0-0 \
        libexpat1 \
        libfontconfig1 \
    && rm -rf /var/lib/apt/lists/*

# Torch family first from the chosen index, then the rest from PyPI.
COPY requirements-core.txt ./
RUN if [ "$TORCH_INDEX" = "cu124" ]; then \
        pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 \
            --index-url https://download.pytorch.org/whl/cu124; \
    else \
        pip install torch==2.6.0 torchvision==0.21.0 \
            --index-url https://download.pytorch.org/whl/cpu; \
    fi \
    && pip install -r requirements-core.txt

# Shipped fine-tuned depth backbone (GAMUS-fine-tuned depth). The base
# Depth-Anything Small weights are pre-cached in the model layer below so the
# image runs offline except for SRTM baseline fetch at job time.
COPY models/finetuned models/finetuned

# Bundled demo samples used by the UI buttons (hilly / forest / urban).
COPY data/external data/external
COPY data/heldout data/heldout

# App + frontend.
COPY app app
COPY depthwizard depthwizard
COPY static static
COPY requirements*.txt ./

# Pre-cache Depth-Anything base weights + fine-tuned backbone (CPU load is fine).
RUN python -c "from depthwizard.depth import DepthEstimator; DepthEstimator(device='cpu')" \
    || echo "model cache skipped (downloads on first job)"

RUN mkdir -p /app/results

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]