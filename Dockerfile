FROM python:3.10

WORKDIR /app

COPY . .

RUN apt-get update && apt-get install -y python3-pip libdbus-1-dev

RUN pip install cupy-cuda12x

RUN pip install \
    --extra-index-url=https://pypi.nvidia.com \
    "cudf-cu12==24.12.*" "dask-cudf-cu12==24.12.*" "cuml-cu12==24.12.*" \
    "cugraph-cu12==24.12.*" "nx-cugraph-cu12==24.12.*" "cuspatial-cu12==24.12.*" \
    "cuproj-cu12==24.12.*" "cuxfilter-cu12==24.12.*" "cucim-cu12==24.12.*" \
    "pylibraft-cu12==24.12.*" "raft-dask-cu12==24.12.*" "cuvs-cu12==24.12.*" \
    "nx-cugraph-cu12==24.12.*"

RUN pip install -r requirements.txt

RUN pip install "fastapi[standard]"

EXPOSE 8000

CMD ["fastapi", "main.py"]
