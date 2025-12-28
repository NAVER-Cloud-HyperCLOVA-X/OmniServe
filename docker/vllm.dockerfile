FROM vllm/vllm-openai:v0.11.2

COPY --chown=appuser:appgroup util/storage/wbl_storage_utility /wbl_storage_utility
RUN pip install --no-cache-dir /wbl_storage_utility

COPY ../vllm-plugins /vllm-workspace/vllm-plugins
RUN pip install -e /vllm-workspace/vllm-plugins

COPY ../vllm /vllm-workspace/vllm
RUN cd /vllm-workspace/ 
RUN diff -ur /usr/local/lib/python3.12/dist-packages/vllm vllm/vllm > vllm.diff || true
RUN cd /usr/local/lib/python3.12/dist-packages/vllm/ && \
    patch -p2 < /vllm-workspace/vllm.diff && \
    cp /vllm-workspace/vllm/vllm/config/reasoning.py /usr/local/lib/python3.12/dist-packages/vllm/config/reasoning.py && \
    cp /vllm-workspace/vllm/vllm/entrypoints/openai/embedding_processor.py /usr/local/lib/python3.12/dist-packages/vllm/entrypoints/openai/embedding_processor.py && \
    cp /vllm-workspace/vllm/vllm/entrypoints/openai/tool_parsers/omni_tool_parser.py /usr/local/lib/python3.12/dist-packages/vllm/entrypoints/openai/tool_parsers/omni_tool_parser.py

EXPOSE 10032
