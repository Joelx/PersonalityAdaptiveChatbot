# from rasa.nlu.components import Component
# from rasa.nlu import utils
# from rasa.nlu.model import Metadata
#
# #import nltk
# #from nltk.sentiment.vader import SentimentIntensityAnalyzer
# import os
#
#
# class GPT3Completion(Component):
#     """A pre-trained sentiment component"""
#
#     name = "sentiment"
#     provides = ["entities"]
#     requires = []
#     defaults = {}
#     language_list = ["de"]
#
#     def __init__(self, component_config=None):
#         super(GPT3Completion, self).__init__(component_config)
#
#     def train(self, training_data, cfg, **kwargs):
#         """Not needed, because the the model is pretrained"""
#         pass
#
#     def convert_to_rasa(self, value, confidence):
#         """Convert model output into the Rasa NLU compatible output format."""
#
#         entity = {"value": value,
#                   "confidence": confidence,
#                   "entity": "sentiment",
#                   "extractor": "sentiment_extractor"}
#
#         return entity
#
#     def process(self, message, **kwargs):
#         """Retrieve the text message, pass it to the classifier
#             and append the prediction results to the message class."""
#         import openai
#
#         openai.api_key = ""
#         openai.Model.list()
#
#         res = openai.Completion.create(
#             model="text-davinci-003",
#             prompt=message.text,
#             max_tokens=7,
#             temperature=0
#         )
#         key, value = max(res.items(), key=lambda x: x[1])
#         entity = self.convert_to_rasa(key, value)
#         message.set("entities", [entity], add_to_output=True)
#
#
#         # sid = SentimentIntensityAnalyzer()
#         # res = sid.polarity_scores(message.text)
#         # key, value = max(res.items(), key=lambda x: x[1])
#         #
#         # entity = self.convert_to_rasa(key, value)
#         #
#         # message.set("entities", [entity], add_to_output=True)
#
#     def persist(self, model_dir):
#         """Pass because a pre-trained model is already persisted"""
#
#         pass