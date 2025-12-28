FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel

RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    vim \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app/track_b

RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir ninja packaging && \
    pip install --no-cache-dir flash-attn --no-build-isolation
    
COPY decoder/vision/track_b/ /app/track_b/
RUN pip install -r requirements.txt && \
    pip install --no-cache-dir uvicorn einops accelerate

COPY util/storage/wbl_storage_utility /wbl_storage_utility
RUN pip install --no-cache-dir /wbl_storage_utility

CMD ["uvicorn", "serve:app", "--host", "0.0.0.0", "--port", "10063", "--log-level", "info"]
