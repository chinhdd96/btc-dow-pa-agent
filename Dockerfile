# BTC Dow/PA Mono-Agent — local/paper 24/7
FROM python:3.12-slim-bookworm
ARG RELEASE_SHA=unknown

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TZ=UTC \
    PORTFOLIO_DATA_DIR=/data \
    RELEASE_SHA=${RELEASE_SHA}

WORKDIR /app

# System deps (certs for HTTPS to Binance/Groq/Telegram)
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates tzdata \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

COPY config.py main.py memory.json sources.json ./
COPY modules/ ./modules/
COPY generated/core_playbook.md generated/daily_playbook.md generated/context.json ./generated/
COPY knowledge/ ./knowledge/
COPY scripts/ ./scripts/

# Runtime state dirs (mount volumes in compose for persistence)
RUN mkdir -p /app/generated \
    && chmod -R a+rwX /app/generated /app/memory.json

# Non-root
RUN useradd -m -u 10001 bot \
    && mkdir -p /data \
    && chown -R bot:bot /app /data
VOLUME ["/data"]
USER bot

CMD ["python", "main.py"]
