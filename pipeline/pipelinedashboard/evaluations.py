from langchain import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import (
    ChatPromptTemplate, 
    MessagesPlaceholder, 
    SystemMessagePromptTemplate, 
    HumanMessagePromptTemplate
)
from langchain.callbacks.base import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from typing import Text
import os 
import re
import socketio
import asyncio

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
    

class Bot2BotConversator:
    def __init__(self):
        self.llm = ChatOpenAI(streaming=True, callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]), verbose=True, temperature=0)
        self.template = """The following is a conversation between a Human and a Chatbot. In this situation, you are the Human. 
        You pretend being a Human that talks to a Chatbot that is designed for psychological counseling and nice chitchat.
        First you are starting with a bit of chitchat. After a bit of verbal exchange you talk about a minor mental issue or stress that you pretend is bothering you.
        
        Again, YOU are the Human in this scenario. The other chatbot doesn't know that you are a Bot!

        Current conversation:
        {history}
        Chatbot: {input}
        Human:"""
        self.prompt = PromptTemplate(
            input_variables=["history", "input"], template=self.template
        )
        self.memory = ConversationBufferMemory(return_messages=True, ai_prefix="Human", human_prefix="Chatbot")
        self.conversation = ConversationChain(memory=self.memory, prompt=self.prompt, llm=self.llm)
        self.MAX_TOKEN_SIZE = 4096
        self.current_tokens = self._count_tokens(self.template)

    def _count_tokens(self, text: Text) -> int:
        tokens = re.findall(r'\S+|\n', text)
        print("------ Token Size of current Prompt ------")
        token_count = len(tokens) - 3 # -2 for our input variables in the system template
        print(token_count)
        return token_count

    def chat(self, message: Text) -> Text:
        response = self.conversation.predict(input=message)
        self.current_tokens += self._count_tokens(message + response)
        return response

class RasaSocketIOBot:
    def __init__(self):
        self.conversator = Bot2BotConversator()
        self.sio = socketio.AsyncClient()
        self.sio.on('bot_uttered', self.on_bot_uttered)

    async def connect(self, api_endpoint: str):
        await self.sio.connect(api_endpoint)

    async def start_conversation(self, start_intent="/start_conversation"):
        await self.sio.emit('user_uttered', {'message': start_intent})

    async def on_bot_uttered(self, data):
        message = data['text']
        response = self.conversator.chat(message)
        print(response)
        await self.sio.emit('bot_uttered', {'message': response})

    async def chat(self, message: Text) -> Text:
        await self.sio.emit('user_uttered', {'message': message})
        await asyncio.sleep(5) # wait for bot to respond
        return self.conversator.chat(message)