FROM python:3.11-slim-bookworm
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
LABEL org.grand-challenge.api-method="exec"
WORKDIR /opt/app
COPY requirements.txt ./
RUN python -m pip install --no-cache-dir -r requirements.txt
RUN useradd -m --no-log-init -r app
COPY inference.py label_schema.json ./
COPY resources/ ./resources/
USER app
ENTRYPOINT ["python","/opt/app/inference.py"]
