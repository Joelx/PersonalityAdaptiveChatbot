import json
import os
import sys

from pydantic import BaseModel

# Set the virtual environment path
venv_path = os.path.expanduser("C:/Users/Joel/.local/pipx/venvs/rasa")
sys.path.insert(0, venv_path)


from ChatbotPipeline import *
import os
from fastapi import FastAPI
from haystack import Pipeline
from haystack.document_stores import InMemoryDocumentStore
import config

from typing import Text, Optional, Any

os.environ["OPENAI_API_KEY"] = config.OPENAI_API_KEY

# create an instance of the FastAPI app
app = FastAPI()

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
feature_normalizer = NormalizerNode(model_path="models/normalizers/feature_normalizer1.2.2.joblib", input="features")
big_five_pipeline.add_node(component=feature_normalizer, name="FeatureNormalizerNode", inputs=["BigFiveFeaturizer.output_1"])
concatenation_node = ConcatenationNode()
big_five_pipeline.add_node(component=concatenation_node, name="ConcatenationNode", inputs=["FeatureNormalizerNode.output_1", "EmbeddingNormalizerNode.output_1"])
feature_selector = BigFiveFeatureSelectionNode(model_paths=sf_model_paths)
big_five_pipeline.add_node(component=feature_selector, name="BigFiveFeatureSelectionNode", inputs=["ConcatenationNode.output_1"])
big_five_classifier = BigFiveClassifierNode(model_paths=cf_model_paths, thresholds=cf_thresholds)
big_five_pipeline.add_node(component=big_five_classifier, name="BigFiveClassifierNode", inputs=["BigFiveFeatureSelectionNode.output_1"])
response_generator = BigFiveResponseGenerator()
big_five_pipeline.add_node(component=response_generator, name="BigFiveResponseGenerator", inputs=["BigFiveClassifierNode.output_1", "Query"])



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
