# Use an official Python runtime as a parent image
FROM python:3.8-slim-buster

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app


RUN apt-get update && apt-get install -y --no-install-recommends \
vim \
libgraphviz-dev \
graphviz  \
&& \
apt-get clean && \
rm -rf /var/lib/apt/lists/*

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port the app runs on
EXPOSE 8001

RUN python -c "import nltk; nltk.download('punkt')"

# Run the command to start the app
CMD ["uvicorn", "main:app", "--proxy-headers", "--host", "0.0.0.0", "--port", "8001", "--reload"]