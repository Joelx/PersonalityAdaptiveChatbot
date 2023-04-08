import json
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from typing import Dict
import requests
import re
from RabbitMQ import send_to_rabbitmq

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
    sender_id = queryParams['sender_id']
    if conversation_history and sender_id:
        conversation_history.append({"event": "system", "sender_id": sender_id})
        print("--------- Conversation History ----------")
        print(conversation_history)

        res = big_five_pipeline.run(query=conversation_history)
        response = {"response": res['response']}
        print("----------------- RES --------------------")
        print(res)
        try:
            big_five_test_results = queryParams['big_five_test_results']
            if big_five_test_results:
                send_to_rabbitmq(json.dumps(big_five_test_results), queue="big_five_test_results", sender_id=sender_id)
        except KeyError:
            pass
    else:
        response = {"response": "Kann noch nichts generieren"}
    
    return response


"""
Test route via GET method.
Used for quick tests during development.
"""
# define a FastAPI endpoint for a Haystack query
@app.get("/test-query")
async def run():
    sender_id = "12345" # Arbitrary tes value
    conversation_history = [
        {"event": "user", "message": "hallo!"},
        {"event": "cleo", "message": "Hallo, wie kann ich dir helfen?"},
        {"event": "user", "message": "Ich möchte plaudern. Bitte erzähle mir etwas über dich!"},
        {"event": "system", "sender_id": sender_id}
    ]
    print("-------- CONVERSATION HISTORY ---------")
    print(conversation_history)
    res = big_five_pipeline.run(query=conversation_history)
    response = {"response": res['response']}    

    big_five_test_results = {
        "neuroticism_test_result": "50",
        "extraversion_test_result": "60",
        "openness_test_result": "70",
        "agreeableness_test_result": "80",
        "conscientiousness_test_result": "90"
    }

    if big_five_test_results:
        send_to_rabbitmq(json.dumps(big_five_test_results), queue="big_five_test_results", sender_id=sender_id)
        print(big_five_test_results)

    return response

