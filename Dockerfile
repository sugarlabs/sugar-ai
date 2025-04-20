# builder
FROM nvidia/cuda:12.8.0-devel-ubuntu22.04 AS builder

WORKDIR /app

COPY requirements.txt .

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3.10 \
        python3.10-dev \
        python3.10-distutils \
        python3-pip \
        build-essential \
        libdbus-1-dev && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir "fastapi[standard]"

# runtime here
FROM nvidia/cuda:12.8.0-runtime-ubuntu22.04

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3.10 \
        python3.10-dev \
        python3-pip \
        libdbus-1-dev \
        build-essential && \
    rm -rf /var/lib/apt/lists/*

COPY --from=builder /usr/local/lib/python3.10 /usr/local/lib/python3.10
COPY --from=builder /usr/local/bin /usr/local/bin

COPY *.py ./
COPY templates/ ./templates/
COPY static/ ./static/
COPY docs/ ./docs/
COPY app/ ./app/
COPY .env* ./
RUN mkdir -p /app/data
EXPOSE 8000
ENV PYTHONUNBUFFERED=1

CMD ["python3", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
