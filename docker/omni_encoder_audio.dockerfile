# Audio Encoder Track B - Omni-Modal Audio (Continuous + Discrete)
FROM nvcr.io/nvidia/pytorch:25.10-py3

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Install system dependencies for audio processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    libsndfile1 \
    libsndfile1-dev \
    ffmpeg \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libswresample-dev \
    sox \
    libsox-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
COPY encoder/audio/track_b .

# Install Python dependencies from requirements.txt
RUN pip install --no-cache-dir -r /workspace/requirements.txt

COPY util/storage/wbl_storage_utility /wbl_storage_utility
RUN pip install --no-cache-dir /wbl_storage_utility

# Expose port
EXPOSE 8002

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8002/health || exit 1

# Run the application
CMD ["uvicorn", "app.server:app", "--host", "0.0.0.0", "--port", "8002", "--log-level", "info"]
