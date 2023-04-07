# Use an official Python runtime as a parent image
FROM python:3.8-slim-buster

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install -r requirements.txt gunicorn uvicorn

# Expose port 8050 for the Dash app
EXPOSE 8050

# Run app.py when the container launches
#CMD ["python", "app.py"]
# Start the server using Uvicorn
#CMD ["uvicorn", "app:server", "--host", "0.0.0.0", "--port", "8050", "--reload"]

CMD ["gunicorn", "app:server", "--bind", "0.0.0.0:8050", "--reload"] 
#CMD ["gunicorn", "app:server", "--bind", "0.0.0.0:8050", "--worker-class", "uvicorn.workers.UvicornWorker"] 

# AUF UVICORN WECHSELN UND keep_alive_timeout=60) einfügen!!