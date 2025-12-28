FROM python:3.13-trixie

# Install ffmpeg for pydub
USER root
RUN apt-get update && \
    apt-get install -y ffmpeg libavcodec-extra && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    groupadd -r appgroup && \
    useradd -m -s /bin/bash -g appgroup appuser
USER appuser

WORKDIR /app
COPY --chown=appuser:appgroup decoder/audio/track_b/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY --chown=appuser:appgroup util/storage/wbl_storage_utility /wbl_storage_utility
RUN pip install --no-cache-dir /wbl_storage_utility

COPY --chown=appuser:appgroup decoder/audio/track_b/app /app/serve

ENTRYPOINT ["python3", "-m", "uvicorn", "--host", "0.0.0.0", "--port", "8000", "serve.main:app"]
