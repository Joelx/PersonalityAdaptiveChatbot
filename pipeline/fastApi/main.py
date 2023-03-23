import panel as pn
from bokeh.embed import server_document
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from typing import Dict
from dashboard.pn_app import createApp
import requests
import re

# Append parent directory
import sys 
sys.path.append('..')
# Import ChatbotPipeline module from parent folder
from ChatbotPipeline import create_pipeline

# create an instance of the FastAPI app
app = FastAPI()
templates = Jinja2Templates(directory="templates")

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
async def bkapp_page(request: Request):
   script = server_document('http://0.0.0.0:4999/app')
   return templates.TemplateResponse("dashboard.html", {"request": request, "script": script})

# @app.get("/dashboard")
# async def bkapp_page(request: Request):
#     script = server_document('http://0.0.0.0:4999/app')
#     bokeh_url = re.search(r"xhr.open\('GET', \"(.*?)\"", script).group(1)
#     print(bokeh_url)
#     bokeh_script = requests.get(bokeh_url).text
#     print(bokeh_script)

#     html_content = f"""<html>
#         <head>
#             <title>Test</title>
#         </head>
#         <body>
#         <h1>Test</h1>
#         <div id="plot"></div>
#         <script>
#             {bokeh_script}
#         </script>
#         </body>
#     </html>
#     """
#     return HTMLResponse(content=html_content)

pn.serve({'/app': createApp},
        port=4999, allow_websocket_origin=["0.0.0.0:8001", "0.0.0.0", "127.0.0.1:8001"],
        address="0.0.0.0", show=False)













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
