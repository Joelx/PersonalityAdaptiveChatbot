import json
from fastapi import FastAPI
from typing import Dict
from RabbitMQ import send_to_rabbitmq
import mlflow

# Append parent directory
import sys 
sys.path.append('..')
# Import ChatbotPipeline module from parent folder
from ChatbotPipeline import create_pipeline

# create an instance of the FastAPI app
app = FastAPI()

big_five_pipeline = create_pipeline()

sender_id_run_map = {}  # A mapping from sender_id to run_id

# Start a unique mlflow run for each sender id. 
# This way we can separate our artifacts.
def start_mlflow_run(session_id):
    if session_id not in sender_id_run_map:
        with mlflow.start_run(nested=True):
            mlflow.set_tag("session_id", session_id)
            run_id = mlflow.active_run().info.run_id
            sender_id_run_map[session_id] = run_id
    else:
        run_id = sender_id_run_map[session_id]
    return run_id

"""
Main route via POST method.
Receives requests from the Chatbot Framework.
"""
# define a FastAPI endpoint for a Haystack query
@app.post("/query")
async def run(queryParams: Dict):
    conversation_history = queryParams['conversation_history'] # Get conversation history from POST 
    sender_id = queryParams['sender_id'] # Get rasa conversation id from POST
    run_id = start_mlflow_run(sender_id) # Start unique mlflow run for this conversation
    if conversation_history and sender_id and run_id:
        # Append run and sender id to the object, so that our pipeline has access
        conversation_history.append({"event": "system", "sender_id": sender_id, "run_id": run_id}) 

        res = big_five_pipeline.run(query=conversation_history) # Run pipeline and get NLG response
        response = {
            "response": res['response'],
            "classification_result": res['predictions'],
            "eval_classification_result": res['eval_predictions']
        }
        try:
            big_five_test_results = queryParams['big_five_test_results'] # Get test results from POST (if exists)
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
    sender_id = "12345" # Arbitrary test value
    run_id = start_mlflow_run(sender_id)
    conversation_history = [
        {"event": "user", "message": "hallo!"},
        {"event": "cleo", "message": "Hallo, wie kann ich dir helfen?"},
        {"event": "user", "message": "Ich möchte plaudern. Bitte erzähle mir etwas über dich!"},
        {"event": "system", "sender_id": sender_id, "run_id": run_id}
    ]
    print("-------- CONVERSATION HISTORY ---------")
    print(conversation_history)
    res = big_five_pipeline.run(query=conversation_history)
    response = {
            "response": res['response'],
            "classification_result": res['predictions'],
            "eval_classification_result": res['eval_predictions']
    }

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

