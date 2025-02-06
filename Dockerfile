FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04 AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
      python3.9 python3-pip build-essential && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

COPY . .

FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04
WORKDIR /app

COPY --from=builder /usr/local/lib/python3.9/site-packages/ /usr/local/lib/python3.9/site-packages/
COPY --from=builder /app /app

CMD ["python3", "main.py"]
