import json
import os
import sys

# Set the virtual environment path
venv_path = os.path.expanduser("C:/Users/Joel/.local/pipx/venvs/rasa")
sys.path.insert(0, venv_path)

import asyncio
import nest_asyncio
from ChatbotPipeline import *
import os
from fastapi import FastAPI
from haystack import Pipeline
from haystack.document_stores import InMemoryDocumentStore
import config

from urllib import parse
from aiohttp import ClientSession

from typing import Text, Optional, Any
#import rasa.shared.utils.cli
#from rasa.shared.core import events
#from rasa.shared.core.training_data.structures import Story
#from rasa.shared.core.training_data.story_writer.yaml_story_writer import (
#    YAMLStoryWriter,
#)

os.environ["OPENAI_API_KEY"] = config.OPENAI_API_KEY

# create an instance of the FastAPI app
app = FastAPI()

# Build in memory document store
"""
document_store = InMemoryDocumentStore()
dicts = [
    {
        'content': 'Email von einer Kollegin. Gestern, gemeinsam mit Freunden beim Grillen. Aufstehen und beginnen zu arbeiten. Alles funktioniert und es macht Spaß. Veränderung des Verhaltens der Menschen durch die Pandemie. Aufgaben verteilen. lustig, ungezwungen',
        'meta': {'name': 'chat_history'}
    },
]
document_store.write_documents(dicts)
"""
sf_model_paths = {
    "neuroticism": "models/feature_selectors/neuroticism_sf_selector2.joblib",
    "extraversion": "models/feature_selectors/extraversion_sf_selector2.joblib",
    "openness": "models/feature_selectors/openness_sf_selector2.joblib",
    "agreeableness": "models/feature_selectors/agreeableness_sf_selector2.joblib",
    "conscientiousness": "models/feature_selectors/conscientiousness_sf_selector2.joblib"
}
cf_model_paths = {
    "neuroticism": "models/big_five_classifiers/neuroticism_classifier2.joblib",
    "extraversion": "models/big_five_classifiers/extraversion_classifier2.joblib",
    "openness": "models/big_five_classifiers/openness_classifier2.joblib",
    "agreeableness": "models/big_five_classifiers/agreeableness_classifier2.joblib",
    "conscientiousness": "models/big_five_classifiers/conscientiousness_classifier2.joblib"
}
cf_thresholds = {
    "neuroticism": 0.578,
    "extraversion": 0.478,
    "openness": 0.178,
    "agreeableness": 0.494,
    "conscientiousness": 0.299
}


big_five_pipeline = Pipeline()
history_retreiver = ConversationHistoryRetreiver()
big_five_pipeline.add_node(component=history_retreiver, name="ConversationHistoryRetreiver", inputs=["Query"])
tfidf_embedding = TfidfVectorizerNode()
big_five_pipeline.add_node(component=tfidf_embedding, name="TfidfVectorizerNode", inputs=["ConversationHistoryRetreiver.output_1"])
fasttext_vectorizer = FasttextVectorizerNode()
big_five_pipeline.add_node(component=fasttext_vectorizer, name="FasttextVectorizerNode", inputs=["TfidfVectorizerNode.output_1"])     
embedding_normalizer = NormalizerNode(input="embeddings")
big_five_pipeline.add_node(component=embedding_normalizer, name="EmbeddingNormalizerNode", inputs=["FasttextVectorizerNode.output_1"])  
featurizer = BigFiveFeaturizer()
big_five_pipeline.add_node(component=featurizer, name="BigFiveFeaturizer", inputs=["ConversationHistoryRetreiver.output_1"]) 
feature_normalizer = NormalizerNode(model_path="models/normalizers/feature_normalizer2.joblib", input="features")
big_five_pipeline.add_node(component=feature_normalizer, name="FeatureNormalizerNode", inputs=["BigFiveFeaturizer.output_1"])
concatenation_node = ConcatenationNode()
big_five_pipeline.add_node(component=concatenation_node, name="ConcatenationNode", inputs=["FeatureNormalizerNode.output_1", "EmbeddingNormalizerNode.output_1"])
feature_selector = BigFiveFeatureSelectionNode(model_paths=sf_model_paths)
big_five_pipeline.add_node(component=feature_selector, name="BigFiveFeatureSelectionNode", inputs=["ConcatenationNode.output_1"])
big_five_classifier = BigFiveClassifierNode(model_paths=cf_model_paths, thresholds=cf_thresholds)
big_five_pipeline.add_node(component=big_five_classifier, name="BigFiveClassifierNode", inputs=["BigFiveFeatureSelectionNode.output_1"])
response_generator = BigFiveResponseGenerator()
big_five_pipeline.add_node(component=response_generator, name="BigFiveResponseGenerator", inputs=["BigFiveClassifierNode.output_1", "Query"])


async def _login(username: Text, password: Text, url: Text) -> Text:
    """Log into Rasa X.
    Args:
        username: Username.
        password: Password of the user.
        url: URL of the Rasa X instance which should logged into.
    Returns:
        The JWT access token of the user.
    """
    url = parse.urljoin(url, "api/auth")
    payload = {"username": username, "password": password}
    async with ClientSession() as session:
        response = await session.post(url, json=payload)
        assert response.status == 200

        response_body = await response.json()
        access_token = response_body["access_token"]
        assert access_token
        return access_token
    
def _client_session(access_token: Text) -> ClientSession:
    headers = {"Authorization": f"Bearer {access_token}"}
    return ClientSession(headers=headers)

def story_to_yaml(story, conversation_id: Text) -> Text:
    """
    Transform a story to YAML.
    Args:
        story: A story.
        conversation_id: An ID of the conversation that was fetched.
    Returns:
        A YAML containing all the story steps.
    """
    print(story)
    return story

    #yaml_writer = YAMLStoryWriter()
    #return yaml_writer.dumps(story.story_steps, is_test_story=True, is_appendable=True)

async def _fetch_full_conversation(
    session: ClientSession, conversation_id: Text
) -> Optional[Dict[Text, Any]]:
    """
    Gets a full conversation from Rasa X API.
    Args:
        session: An initialized client session.
        conversation_id: An ID of the conversation to fetch.
    Returns:
        A full conversation for a specified `conversation_id`.
    """
    url = parse.urljoin(config.RASA_X_URL, "api/conversations/" + conversation_id)
    response = await session.get(url)
    if response.status != 200:
        #rasa.shared.utils.cli.print_warning(f"Unable to call GET {url}.")
        return None
    return await response.json()

async def _fetch_conversations(session: ClientSession) -> List[Dict[Text, Any]]:
    """Gets conversations from Rasa X API.
    Args:
        session: An initialized client session.
    Returns:
        A list of all conversations.
    """
    url = parse.urljoin(config.RASA_X_URL, "api/conversations")
    response = await session.get(url)
    if response.status != 200:
        #rasa.shared.utils.cli.print_error(f"Unable to call GET {url}.")
        return []
    return await response.json()

async def fetch_rasa_api_conversation(
    conversation_id: Text
) -> Text:
    """Fetch story from running Rasa X instance by conversation ID.
    Args:
        conversation_id: An ID of the conversation to fetch.
    Returns:
        Extracted story in json format.
    """
    access_token = await _login(config.RASA_X_USERNAME, config.RASA_X_PASSWORD, config.RASA_X_URL)
    conversation = []
    async with _client_session(access_token) as session:
        conversation_data = await _fetch_full_conversation(session, conversation_id)
        
        if conversation_data:
            conversation_map = {"user": "user", "bot": "cleo"}
            for event in conversation_data['events']:
                conversation_type = conversation_map.get(event["event"])
                if conversation_type:
                    conversation.append({"event": conversation_type, "message": event["text"]})    
    return conversation

# define a FastAPI endpoint for a Haystack query
@app.get("/query/{input}")
async def run(input: str):
    nest_asyncio.apply()
    loop = asyncio.get_event_loop()
    conversation_history = loop.run_until_complete(
            fetch_rasa_api_conversation("2e14fc1bf1f64800b53a3244946e4905")
        )
    print(conversation_history)
    res = big_five_pipeline.run(query=conversation_history)
    response = {"response": res['response']}
    print("----------------- RES --------------------")
    print(res)
    #big_five_pipeline.draw()
    return response
