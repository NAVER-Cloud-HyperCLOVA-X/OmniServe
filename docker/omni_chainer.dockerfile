FROM python:3.12-alpine3.21

# Create non-root user
RUN addgroup -S appgroup && adduser -S appuser -G appgroup

WORKDIR /app

COPY ../omni_chainer/pyproject.toml ./
COPY ../omni_chainer/omni_chainer ./omni_chainer
COPY --chown=appuser:appgroup util/storage/wbl_storage_utility /wbl_storage_utility

RUN pip install --no-cache-dir /wbl_storage_utility && \
    pip install --no-cache-dir -e .
ENV PATH="/home/appuser/.local/bin:${PATH}"

EXPOSE 8000

USER appuser

CMD ["fastapi", "run", "--host", "0.0.0.0", "--port", "8000", "omni_chainer/main.py"]
