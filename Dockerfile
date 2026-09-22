FROM nvidia/cuda:12.8.0-devel-ubuntu22.04 AS builder

WORKDIR /app

COPY requirements.txt ./
COPY requirements/ ./requirements/

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3.10 \
        python3.10-dev \
        python3.10-distutils \
        python3-pip \
        build-essential \
        libdbus-1-dev && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /tmp/wheels && \
    for attempt in 1 2 3 4 5 6 7 8 9 10; do \
        if pip download --dest /tmp/wheels -r requirements/cuda.txt; then \
            break; \
        fi; \
        if [ "$attempt" = "10" ]; then \
            exit 1; \
        fi; \
        echo "Dependency download attempt $attempt failed; retrying..." >&2; \
    done && \
    pip install --no-index --find-links=/tmp/wheels \
        -r requirements/cuda.txt


FROM nvidia/cuda:12.8.0-runtime-ubuntu22.04 AS runtime

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

RUN mkdir -p /app/data

EXPOSE 8000
ENV PYTHONUNBUFFERED=1

CMD ["python3", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
