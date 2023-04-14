from typing import Dict, Text, Any, List

from rasa.engine.graph import GraphComponent, ExecutionContext
from rasa.engine.recipes.default_recipe import DefaultV1Recipe
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.shared.nlu.training_data.message import Message
from rasa.shared.nlu.training_data.training_data import TrainingData

# TODO: Correctly register your component with its type
@DefaultV1Recipe.register([DefaultV1Recipe.ComponentType.MESSAGE_TOKENIZER], is_trainable=False)

class GPT3Completion(GraphComponent):
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


    def process(self, messages: List[Message]) -> List[Message]:
        # TODO: This is the method which Rasa Open Source will call during inference.
        import openai
        print(TrainingData)
        print(TrainingData.nlu_examples)
        print(TrainingData.response_examples)
        openai.api_key = "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"
        #openai.Model.list()
        for message in messages:
            if 'text' in message.data.keys():
                res = openai.Completion.create(
                    model="text-davinci-003",
                    prompt=message.data["text"],
                    max_tokens=200,
                    temperature=0
                )
                print(res["choices"][0]["text"])
        #message.set("entities", [entity], add_to_output=True)
        # from revChatGPT.revChatGPT import Chatbot
        #
        # config = {
        #     "email": "schlotthauerjo71188@th-nuernberg.de",
        #     "password": "kFKqDxNBrtnZH35"  # ,
        #     # "session_token": "<SESSION_TOKEN>", # Deprecated. Use only if you encounter captcha with email/password
        #     # "proxy": "<HTTP/HTTPS_PROXY>"
        # }
        #
        # chatbot = Chatbot(config, conversation_id=None)

        return messages

