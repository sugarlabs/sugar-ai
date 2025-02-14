FROM python:3.10-slim

WORKDIR /app

COPY . .

RUN apt-get update && \
    apt-get install -y --no-install-recommends python3-pip libdbus-1-dev && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir cupy-cuda12x

RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir "fastapi[standard]"

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
