import json
from langchain import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from typing import Text
import re
import socketio
from RabbitMQ import *

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
        self.llm = ChatOpenAI(streaming=False, temperature=0)
        #self.llm = ChatOpenAI(streaming=True, callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]), verbose=True, temperature=0)
        # self.template = """You are in a conversation with another bot. You will get utterance by utterance from the other bot.
        # You pretend being a Human that talks to a Chatbot that is designed for psychological counseling and nice chitchat.
        # First you are starting with a bit of chitchat. After a bit of verbal exchange you talk about a minor mental issue or stress that you pretend is bothering you.
        # The Chatbot will try and detect your personality to adapt to your needs. So try to immitate a persona e.g. personality that is indidacted via a Big Five Personality Model Profile.
        # Try to challenge the chatbot to adapt to you by getting to know your Big Five personality persona.

        # Again, YOU are the Human in this scenario and you are talking GERMAN! The other chatbot doesn't know that you are a Bot!
        
        # Current conversation:
        # {history}
        # Chatbot: {input}
        # Human:"""
        self.template = """You are in a conversation with another bot. You will get utterance by utterance from the other bot.
        Again, YOU are the Human in this scenario, NOT the bot and you are talking GERMAN! The other chatbot doesn't know that you are a Bot!
        
        Current conversation:
        {history}
        Chatbot: {input}
        Human:"""
        self.prompt = PromptTemplate(
            input_variables=["history", "input"], template=self.template
        )
        self.memory = ConversationBufferMemory(return_messages=True, ai_prefix="Human", human_prefix="Chatbot")
        self.conversation = ConversationChain(memory=self.memory, prompt=self.prompt, llm=self.llm)
        self.MAX_TOKEN_SIZE = 1024 # Arbitraty amount smaller than 4096. We don't want the conversations to be too long.
        self.current_tokens = self._count_tokens(self.template)

    def _count_tokens(self, text: Text) -> int:
        tokens = re.findall(r'\S+|\n', text)
        #print("------ Token Size of current Prompt ------")
        token_count = len(tokens) - 3 # -2 for our input variables in the system template
        #print(token_count)
        return token_count

    def chat(self, message: Text) -> Text:
        #print(f"Conversator received: {message}")
        response = self.conversation.predict(input=message)
        #print(f"Conversator produced: {response}")
        self.current_tokens += self._count_tokens(message + response)
        return response


# class RasaSocketIOBot:
#     def __init__(self):
#         self.conversator = Bot2BotConversator()
#         self.sio = socketio.AsyncClient()
#         self.sio.on('bot_uttered', self.on_bot_uttered)
#         self.UUID = str(uuid.uuid4().hex)[:32]
#         self.message = {
#             "sender_id": self.UUID,
#             "timestamp": round(time.time() * 1000) / 1000,
#             "text": "",
#             "metadata": {
#                 "platform": "socketio-evaluation",
#                 "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:123.0) Gecko/20100101 Firefox/123.0",
#                 "language": "de-DE",
#                 "timezone": "UTC+1",
#             },
#             "input_channel": "socketio-evaluation",
#             "room": self.UUID
#         }

#     async def connect(self, api_endpoint: str, socket_path: str):
#         await self.sio.connect(api_endpoint, socketio_path=socket_path, transports='websocket')
#         print('Connected! SID is', self.sio.sid)

#     async def start_conversation(self, start_intent="/start_conversation"):
#         self.message["text"] = start_intent
#         await self.sio.emit('user_uttered', self.message)
#         print("finished start")
#         await self.on_bot_uttered({'text': 'Hallo!'})

#     async def on_bot_uttered(self, data):
#         print("On bot uttered!")
#         message = data['text']
#         response = self.conversator.chat(message)
#         print(response)
#         return response

#     async def chat(self, message: Text) -> Text:
#         print("Chat!")
#         self.message["text"] = message
#         await self.sio.emit('user_uttered', self.message)
#         response = await self.sio.wait_for_event('bot_uttered')
#         return response['text']






class RasaChatClient:
    def __init__(self, server_url, socketio_path, sender_id):
        self.conversator = Bot2BotConversator()
        self.server_url = server_url
        self.socketio_path = socketio_path
        self.sender_id = sender_id
        self.sio = socketio.Client()
        self.sio.on('connect', self.on_connect)
        self.sio.on('connect_error', self.on_connect_error)
        self.sio.on('bot_uttered', self.on_bot_uttered)

    def connect(self):
        self.sio.connect(self.server_url, socketio_path=self.socketio_path, transports="websocket")

    def disconnect(self):
        self.sio.disconnect()

    def on_connect(self):
        print('Connected to Socket.io server.')
        self.sio.emit('session_request', {'session_id': self.get_session_id()})
        print(f"Session ID: {self.get_session_id()}")

    def on_connect_error(self, error):
        # Write any connection errors to the console
        print(error)

    def on_bot_uttered(self, response):
        #print('Bot uttered:', response)
        if 'text' in response:
            print(f"Chatbot: {response['text']}")
            send_to_rabbitmq(json.dumps({"ai_bot_uttered": response['text']}), queue="eval_bot_message", sender_id=self.sender_id)
            self.generate_conversator_response(response['text'])

    def generate_conversator_response(self, message):
        #print("Called conversation generator")
        conversator_response = self.conversator.chat(message)
        #print(f"Human-Bot: {conversator_response}")
        send_to_rabbitmq(json.dumps({"human_bot_uttered": conversator_response}), queue="eval_bot_message", sender_id=self.sender_id)
        self.utter(conversator_response)
        if self.conversator.current_tokens >= self.conversator.MAX_TOKEN_SIZE:
            print("Max token size exceeded. Disconnecting...")
            self.disconnect()

    def utter(self, msg):
        self.sio.emit('user_uttered', {'message': msg, 'session_id': self.get_session_id()})

    def get_session_id(self):
        # You can use any method to store and retrieve the session ID here, such as a file or a database
        # This example uses a dictionary to store the ID in memory
        if self.sio.eio.sid:
            return self.sio.eio.sid
        new_id = self.sio.sid
        if new_id:
            self.sio.eio.sid['RASA_SESSION_ID'] = new_id
        return new_id

    #def append_message(self, message, message_type):
    #    print(f"Chatbot: {message}")
        # Define your own implementation of this function to handle the message received from the RASA bot
    #    pass







# class RasaSocketIOBot:
#     def __init__(self):
#         self.conversator = Bot2BotConversator()
#         self.sio = socketio.AsyncClient()
#         self.sio.on('bot_uttered', self.on_bot_uttered)
#         self.UUID = str(uuid.uuid4().hex)[:32]
#         self.message = {
#             "sender": self.UUID,
#             "timestamp": round(time.time() * 1000) / 1000,
#             "message": "",
#             "metadata": {
#                 "platform": "socketio-evaluation",
#                 "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:123.0) Gecko/20100101 Firefox/123.0",
#                 "language": "de-DE",
#                 "timezone": "UTC+1",
#             },
#             "input_channel": "socketio-evaluation",
#             "room": self.UUID
#         }

#     async def connect(self, api_endpoint: str, socket_path: str):
#         await self.sio.connect(api_endpoint, socketio_path=socket_path, transports='websocket')
#         print('Connected! SID is', self.sio.sid)

#     async def start_conversation(self, start_intent="/start_conversation"):
#         self.message["message"] = start_intent
#         await self.sio.emit('user_uttered', self.message)
#         print("Sent start conversation utterance")
#         #await self.chat(start_intent)

#     async def on_bot_uttered(self, data):
#         print("On bot uttered!")
#         message = data['message']
#         response = self.conversator.chat(message)
#         print("Generated response:", response)
#         self.message["message"] = response
#         await self.sio.emit('user_uttered', self.message)
#         print("Sent generated response to Rasa")

#     async def chat(self, message: Text) -> Text:
#         await self.start_conversation()
#         while True:
#             response_received = asyncio.Event()

#             def on_bot_uttered(data):
#                 print("On bot uttered!")
#                 message = data['message']
#                 response = self.conversator.chat(message)
#                 print("Generated response:", response)
#                 self.message["message"] = response
#                 self.sio.emit('user_uttered', self.message)
#                 print("Sent generated response to Rasa")
#                 response_received.set()

#             self.sio.on('bot_uttered', on_bot_uttered)

#             self.message["message"] = message
#             await self.sio.emit('user_uttered', self.message)
#             print("Sent user utterance to Rasa")

#             await response_received.wait()
#             self.sio.off('bot_uttered', on_bot_uttered)

#             # Wait a bit to allow the bot's response to be sent
#             await asyncio.sleep(8)
    
# async def async_socketio_test():
#     sio = socketio.AsyncClient()
#     @sio.event
#     async def connect():
#         print('connection established')
#     @sio.event
#     async def my_message(data):
#         print('message received with ', data)
#         await sio.emit('user_uttered', {'message': "Hallo"})
#     @sio.event
#     async def disconnect():
#         print('disconnected from server')

#     await sio.connect('https://joel-schlotthauer.com', socketio_path='/rasax/socket.io/')
#     #await sio.connect('joel-schlotthauer.com?path=/rasax/socket.io/')
#     await sio.wait()
    