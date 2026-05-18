FROM ghcr.io/berriai/litellm:v1.83.14-stable.patch.3

COPY --from=ghcr.io/astral-sh/uv:0.5.11 /uv /usr/local/bin/uv

RUN uv pip install --no-cache-dir --python /app/.venv/bin/python transformers tokenizers

ENV HF_HOME=/opt/hf-cache
RUN mkdir -p /opt/hf-cache && chmod -R a+rX /opt/hf-cache

RUN /app/.venv/bin/python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('nomic-ai/nomic-embed-text-v1')"
RUN /app/.venv/bin/python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('nomic-ai/modernbert-embed-base')"
