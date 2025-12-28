# Vision Encoder Track A - Continuous Vision Model (Qwen2.5-VL)
FROM nvcr.io/nvidia/pytorch:25.10-py3

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Install system dependencies for vision processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY ../encoder/vision/track_a .
COPY --chown=appuser:appgroup util/storage/wbl_storage_utility /wbl_storage_utility

RUN pip install --no-cache-dir /wbl_storage_utility && \
    pip install --no-cache-dir -r requirements.txt

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["uvicorn", "app.server:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "info"]
