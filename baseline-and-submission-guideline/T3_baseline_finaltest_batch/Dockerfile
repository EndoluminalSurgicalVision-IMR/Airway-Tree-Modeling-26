FROM --platform=linux/amd64 python:3.11-slim-bookworm AS runtime

ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu118
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HOME=/tmp/huggingface

LABEL org.atm26.batch="1" org.grand-challenge.api-method="exec" \
      org.atm26.track="3" \
      org.atm26.interface="png-sequences-to-frame-label-csv-v1"

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
RUN python -m pip install --no-cache-dir torch==2.6.0 torchvision==0.21.0 --index-url ${TORCH_INDEX_URL}
WORKDIR /opt/app
COPY requirements.txt ./
RUN python -m pip install --no-cache-dir -r requirements.txt
RUN groupadd -r app && useradd -m --no-log-init -r -g app app
COPY --chown=app:app vendor/ ./vendor/
COPY --chown=app:app runtime.py inference.py core.py label_schema.json ./
USER app
ENTRYPOINT ["python", "/opt/app/inference.py"]

# --target runtime creates a code-only image with a read-only model volume.
# The default image embeds the explicitly staged offline model resources.
FROM runtime AS bundled
COPY --chown=app:app resources/ /opt/app/resources/
