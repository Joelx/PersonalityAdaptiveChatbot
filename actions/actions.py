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

class ActionHaystack(Action):

    def name(self) -> Text:
        return "call_haystack"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        conversation = self._parse_rasa_events_to_conversation(tracker.events)
        url = "http://localhost:8001/query"

        payload = {"conversation_history": conversation}
        print(payload)

        headers = {
            'Content-Type': 'application/json'
        }
        response = requests.request("POST", url, headers=headers, json=payload).json()

        if response["response"]:
            answer = response["response"]
        else:
            answer = "Tut mir leid, ich habe gerade technische Probleme!"

        dispatcher.utter_message(text=answer)

        return []
    
    def _parse_rasa_events_to_conversation(self, events: Dict) -> Text:
        """Fetch story from running Rasa X instance by conversation ID.
        Args:
            conversation_id: An ID of the conversation to fetch.
        Returns:
            Extracted story in json format.
        """
        conversation = []
        if events:
            conversation_map = {"user": "user", "bot": "cleo"}
            for event in events:
                conversation_type = conversation_map.get(event["event"])
                if conversation_type:
                    conversation.append({"event": conversation_type, "message": event["text"]})    
        return conversation
        