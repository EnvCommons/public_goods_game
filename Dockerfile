FROM python:3.12-slim
RUN apt-get update && apt-get install -y curl git && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY env.py server.py ./
EXPOSE 8000
CMD ["python", "server.py"]
