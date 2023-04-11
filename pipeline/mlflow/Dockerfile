# Dockerfile
FROM python:3.8-slim

# Set the working directory to /app
#WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
vim \
&& \
apt-get clean && \
rm -rf /var/lib/apt/lists/*

# Install the necessary packages
RUN pip install mlflow==2.2.2

# Copy the current directory contents into the container at /app
#COPY . /app
#COPY mlflow.db /mlflow.db

# Expose the desired port
EXPOSE 8003

# Start the MLflow server
CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "8003", "--backend-store-uri", "sqlite:///models/mlflow.db", "--default-artifact-root", "/models/artifacts"]