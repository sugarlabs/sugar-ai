FROM python:3.10

WORKDIR /app

COPY . .

RUN apt-get update && apt-get install -y python3-pip libdbus-1-dev

RUN pip install cupy-cuda12x

RUN pip install -r requirements.txt

RUN pip install "fastapi[standard]"

EXPOSE 8000

CMD ["fastapi", "main.py"]
