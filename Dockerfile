FROM nvidia/cuda:12.1.0-devel-ubuntu22.04 
# this base image is for CUDA 12.1.0 and Ubuntu 22.04, works fine

WORKDIR /app

COPY . .

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3.10 \
        python3.10-dev \
        python3.10-distutils \
        python3-pip \
        build-essential \
        libdbus-1-dev && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir cupy-cuda12x
# we can remvove this if we don't wanna use gpu

RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir "fastapi[standard]"

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
