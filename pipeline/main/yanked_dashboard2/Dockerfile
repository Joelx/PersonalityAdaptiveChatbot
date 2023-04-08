FROM python:3.9-slim-buster

WORKDIR /app

# Install necessary dependencies
RUN apt-get update && apt-get install -y \
    libfontconfig1 \
    libfreetype6 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libxrender1 \
    libxtst6 \
    curl

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the app code to the container
COPY app.py .

# Expose the necessary port
EXPOSE 5006

# Start the Bokeh server with your Panel app
CMD ["bokeh", "serve", "--show", "app.py"]