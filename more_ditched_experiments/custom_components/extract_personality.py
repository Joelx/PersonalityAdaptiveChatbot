from typing import Dict, Text, Any, List

from rasa.engine.graph import GraphComponent, ExecutionContext
from rasa.engine.recipes.default_recipe import DefaultV1Recipe
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.shared.nlu.training_data.message import Message
from rasa.shared.nlu.training_data.training_data import TrainingData
import pickle
import numpy as np
import pandas as pd
from featurization import *

# TODO: Correctly register your component with its type
@DefaultV1Recipe.register([DefaultV1Recipe.ComponentType.ENTITY_EXTRACTOR], is_trainable=False)

class ExtractPersonality(GraphComponent):
    @classmethod
    def create(
            cls,
            config: Dict[Text, Any],
            model_storage: ModelStorage,
            resource: Resource,
            execution_context: ExecutionContext,
    ) -> GraphComponent:
        # TODO: Implement this
        ...

    def train(self, training_data: TrainingData) -> Resource:
        # TODO: Implement this if your component requires training
        ...

    def process_training_data(self, training_data: TrainingData) -> TrainingData:
        # TODO: Implement this if your component augments the training data with
        #       tokens or message features which are used by other components
        #       during training.
        ...
        return training_data

    def convert_to_rasa(self, value, confidence):
        """Convert model output into the Rasa NLU compatible output format."""

        entity = {"value": value,
                  "probability": confidence,
                  "entity": "neuroticism",
                  "extractor": "neuroticism_extractor"}

        return entity

    def process(self, messages: List[Message]) -> List[Message]:
        """Retrieve the text message, pass it to the classifier
            and append the prediction results to the message class."""
        import pickle
        # Load the Model back from file
        neuro_model_file = "..\\saved_models\\neuro_model.pkl"
        with open(neuro_model_file, 'rb') as file:
            neuro_model = pickle.load(file)

        selector_model_file = "..\\saved_models\\neuro_selector_model.pkl"
        with open(selector_model_file, 'rb') as file:
            selector = pickle.load(file)

        for message in messages:
            if 'text' in message.data.keys():
                msg_body = [message.data["text"], "dummy"]
                query_df = pd.DataFrame()
                query_df["text"] = np.asarray(msg_body)
                train_features, test_features, feature_names = featurize(query_df, query_df, 'tfidf_glove')
                final_data = selector.transform(train_features.tocsr())

                predict_neuro_proba = neuro_model.predict_proba(final_data)
                #entity = self.convert_to_rasa("neuro", predict_neuro_proba[0][0])
                entity = {"value": "neuro",
                          "probability": predict_neuro_proba[0][0],
                          "entity": "neuroticism",
                          "extractor": "neuroticism_extractor"}

                message.set("entities", [entity], add_to_output=True)
                print(entity)

        return messages
