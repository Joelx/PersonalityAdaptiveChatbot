from langchain import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from typing import Text
import os 

class BigFiveClassificationEvaluator:
    def __init__(self):
        self.llm = ChatOpenAI(temperature=0)
        self.template = """You are a powerful AI model for text classification and with comprehensive knowledge of the Big Five personality model. 
        This knowledge is backed up by scientific research and literature that describes the five traits of the Big Five model and their impact on peoples 
        individual writing style through linguistic cues. 
        You will receive a text written by a human in German language (German Human Text)  that was received from a conversation by that human with a chatbot. You will do a Big Five classification of that text on all five Traits, that will look like that:
        Neuroticism: CF[cf_class], Proba[cf_proba]
        Extraversion: CF[cf_class], Proba[cf_proba]
        Openness for experience: CF[cf_class], Proba[cf_proba]
        Agreeableness: CF[cf_class], Proba[cf_proba]
        Conscientiousness: CF[cf_class], Proba[cf_proba]
        
        - cf_class is be the classification result on each individual trait and will be either 1 (if classification is positive) or 0 (if classification is negative). 
        You will do this classification based on your knowledge about the Big Five traits and their linguistic cues. However, in general a classification is considered 1 (positive) if this trait seems to be pronounced above average in that person.
        - cf_proba is the probability of that classification result on each individual trait and will be a floating point number between 0 and 1. 

        German Human Text:
        {text}
        Do the classification:"""
        self.prompt = PromptTemplate(input_variables=["text"], template=self.template)

    def classify(self, text: Text) -> Text:
        llm_chain = LLMChain(prompt=self.prompt, llm=self.llm)
        cf_result = llm_chain.run(text=text)
        return cf_result