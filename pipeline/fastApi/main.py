import panel as pn
from bokeh.embed import server_document
from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from typing import Dict

# Append parent directory
import sys 
sys.path.append('..')
# Import ChatbotPipeline module from parent folder
from ChatbotPipeline import create_pipeline

# create an instance of the FastAPI app
app = FastAPI()

big_five_pipeline = create_pipeline()

# define a FastAPI endpoint for a Haystack query
@app.post("/query")
async def run(queryParams: Dict):
    conversation_history = queryParams['conversation_history']

    if conversation_history:
        print("--------- Conversation History ----------")
        print(conversation_history)

        res = big_five_pipeline.run(query=conversation_history)
        response = {"response": res['response']}
        print("----------------- RES --------------------")
        print(res)
    else:
        response = {"response": "Kann noch nichts generieren"}
    return response

# define a FastAPI endpoint for our Panel dashboard
@app.get("/dashboard")
async def dashboard():
    # create your panel dashboard
    pn.extension()
    # ...
    return pn.pane.Markdown("# My Panel Dashboard")

"""
Test route via get
"""
# define a FastAPI endpoint for a Haystack query
@app.get("/test-query")
async def run():
    conversation_history = [
        {"event": "user", "message": "hallo!"},
        {"event": "cleo", "message": "Hallo, wie kann ich dir helfen?"}
    ]
    res = big_five_pipeline.run(query=conversation_history)
    response = {"response": res['response']}
    return response
