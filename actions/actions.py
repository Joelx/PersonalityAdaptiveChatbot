# This files contains your custom actions which can be used to run
# custom Python code.
#
# See this guide on how to implement these action:
# https://rasa.com/docs/rasa/custom-actions


# This is a simple example for a custom action which utters "Hello World!"

from typing import Any, Text, Dict, List
#
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
import requests
#
#
class ActionHaystack(Action):

    def name(self) -> Text:
        return "call_haystack"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        url = "http://localhost:8001/query"
        #payload = {"query": str(tracker.latest_message["text"])}
        payload = {"conversation_id": str(tracker.sender_id)}
        print(payload)
        headers = {
            'Content-Type': 'application/json'
        }
        response = requests.request("POST", url, headers=headers, json=payload).json()

        if response["response"]:
            answer = response["response"]
        else:
            answer = "Tut mir leid, ich habe gerade technische Probleme!"

        print(answer)
        dispatcher.utter_message(text=answer)

        return []
