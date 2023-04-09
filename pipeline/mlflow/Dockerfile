# Dockerfile
FROM python:3.8-slim

RUN pip install mlflow

# Copy the mlruns directory and the mlflow.db file into the Docker image
COPY artifacts /artifacts
COPY mlflow.db /mlflow.db

EXPOSE 8003

CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "8003", "--backend-store-uri", "sqlite:///mlflow.db", "--default-artifact-root", "./artifacts"]
