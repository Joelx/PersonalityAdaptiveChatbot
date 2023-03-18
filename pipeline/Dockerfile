version: "3"
services:
haystack-api:
   build:
   context: .
dockerfile: Dockerfile
ports:
- 8000:8000
volumes:
   # Folder for mounting fine-tuned mode
   - "./models:/home/user/models"
environment:
- DB_HOST=elasticsearch
- USE_GPU=True
- TOP_K_PER_SAMPLE=3 # how many answers can come from the same small passage (reduce value for more variety of answers)
restart: always
depends_on:
- elasticsearch
command: "/bin/bash -c 'sleep 50 && gunicorn rest_api.application:app -b 0.0.0.0 -k uvicorn.workers.Uvicorn Worker --workers 1 --timeout 180 --preload'"
elasticsearch:
# Load the new elastic search image with closed domain question-answer documents
image: "q-and-a:version1"
ports:
- 9200:9200
environment:
- discovery.type=single-node
ui:
# For Testing the API
image: "deepset/haystack-streamlit-ui"
ports:
- 8501:8501
environment:
- API_ENDPOINT=http://haystack-api:8000