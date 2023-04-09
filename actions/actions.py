import re
from typing import Any, Text, Dict, List
from rasa_sdk import Action, Tracker, FormValidationAction
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.types import DomainDict
from rasa_sdk.events import SlotSet
import requests
from .neoffi import ComputeNeoFFI as neoffi
#from .neoffi import Spiderchart as spiderchart
#from .neoffi import CreatePDF

class ActionHaystack(Action):
    def __init__(self):
        self.url = "https://joel-schlotthauer.com/pipeline/query"
        #self.url = "http://localhost:8001/query" # Local test URL

    def name(self) -> Text:
        return "call_haystack"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        # Check if a form is running. If so, we want rasa to handle it without junking up our conversation data.
        active_loop = tracker.active_loop.get('name')
        if active_loop:
            print(active_loop)
            return
        
        conversation = self._parse_rasa_events_to_conversation(tracker.events)

        payload = {"conversation_history": conversation}

        # Submit sender(conversation)-id so that we can identify the receipient
        payload['sender_id'] = tracker.sender_id

        # Send over Big Five Test results, if existant
        # All the 5 slots get set simultaniously, so if neuroticism slot does exist, the others do as well
        try:
            big_five_test_results = {
                "neuroticism_test_result": tracker.get_slot('neuroticism_test_result'),
                "extraversion_test_result": tracker.get_slot('extraversion_test_result'),
                "openness_test_result": tracker.get_slot('openness_test_result'),
                "agreeableness_test_result": tracker.get_slot('agreeableness_test_result'),
                "conscientiousness_test_result": tracker.get_slot('conscientiousness_test_result')
            }

            payload["big_five_test_results"] = big_five_test_results
        except:
            pass
        
        print(payload)

        headers = {
            'Content-Type': 'application/json'
        }

        response = []
        try:
            response = requests.request("POST", self.url, headers=headers, json=payload).json()
        except requests.exceptions.HTTPError as err:
            print ("Http Error:", err)

        print(response)
        if response["classification_result"]:
            classification_results = response["classification_result"]
        else: 
            classification_results = "No prediction could be fetched yet"

        if response["eval_classification_result"]:
            eval_classification_results = response["eval_classification_result"]
        else: 
            eval_classification_results = "No evaluation prediction could be fetched yet"
        
        if response["response"]:
            answer = response["response"]
        else:
            answer = "Tut mir leid, ich habe gerade technische Probleme!"

        dispatcher.utter_message(text=answer)

        return [
            SlotSet("classification_results", classification_results),
            SlotSet("eval_classification_results", eval_classification_results)
        ]
    
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
                    # If its only a number (e.g. from Big 5 test), we dont want it
                    if event["text"].isdigit():
                        continue
                    # Likewise we want to filter out potential Big Five questions that have been asked
                    if re.match("Aussage \d+/\d+", event["text"]):
                        continue
                    conversation.append({"event": conversation_type, "message": event["text"]})    
        return conversation
    

class ValidateTestForm(FormValidationAction):
    def name(self) -> Text:
        return "validate_test_form"
    
    def __init__(self):
        self.NEO_FFI_ALLOWED_ANSWERS = ["1", "2", "3", "4", "5", "starke Zustimmung", "starke Ablehnung"]
        self.ITEM_VALIDATION_TEXT = "Erlaubt sind nur die Ganzzahlen 1 bis 5. Nutze am besten einfach die Buttons. Ich stelle die Frage einfach nochmal :)"
    
    def validate_item(self, slot_name: str, slot_value: Any, dispatcher: CollectingDispatcher, tracker: Tracker, domain: DomainDict) -> Dict[Text, Any]:
        if slot_value not in self.NEO_FFI_ALLOWED_ANSWERS:
            dispatcher.utter_message(self.ITEM_VALIDATION_TEXT)
            return {slot_name: None}
        else:
            if slot_value == "starke Zustimmung":
                slot_value = 5
            elif slot_value == "starke Ablehnung":
                slot_value = 1
            print(slot_value)
            return {slot_name: slot_value}
        
    
    def validate_neu_item_6(self, slot_value: Any, dispatcher: CollectingDispatcher, tracker: Tracker, domain: DomainDict) -> Dict[Text, Any]:
        return self.validate_item("neu_item_6", slot_value, dispatcher, tracker, domain)

    def validate_neu_item_21(self, slot_value: Any, dispatcher: CollectingDispatcher, tracker: Tracker, domain: DomainDict) -> Dict[Text, Any]:
        return self.validate_item("neu_item_21", slot_value, dispatcher, tracker, domain)

    def validate_neu_item_11(self, slot_value: Any, dispatcher: CollectingDispatcher, tracker: Tracker, domain: DomainDict) -> Dict[Text, Any]:
        return self.validate_item("neu_item_11", slot_value, dispatcher, tracker, domain)

    def validate_neu_item_26(self, slot_value: Any, dispatcher: CollectingDispatcher, tracker: Tracker, domain: DomainDict) -> Dict[Text, Any]:
        return self.validate_item("neu_item_26", slot_value, dispatcher, tracker, domain)

    def validate_neu_item_41(self, slot_value: Any, dispatcher: CollectingDispatcher, tracker: Tracker, domain: DomainDict) -> Dict[Text, Any]:
        return self.validate_item("neu_item_41", slot_value, dispatcher, tracker, domain)

    def validate_neu_item_51(self, slot_value: Any, dispatcher: CollectingDispatcher, tracker: Tracker, domain: DomainDict) -> Dict[Text, Any]:
        return self.validate_item("neu_item_51", slot_value, dispatcher, tracker, domain)

    def validate_ext_item_2(self, slot_value: Any, dispatcher: CollectingDispatcher, tracker: Tracker, domain: DomainDict) -> Dict[Text, Any]:
        print("validating")
        return self.validate_item("ext_item_2", slot_value, dispatcher, tracker, domain)

    def validate_ext_item_7(self, slot_value: Any, dispatcher: CollectingDispatcher, tracker: Tracker, domain: DomainDict) -> Dict[Text, Any]:
        return self.validate_item("ext_item_7", slot_value, dispatcher, tracker, domain)

    def validate_ext_item_22(self, slot_value: Any, dispatcher: CollectingDispatcher, tracker: Tracker, domain: DomainDict) -> Dict[Text, Any]:
        return self.validate_item("ext_item_22", slot_value, dispatcher, tracker, domain)

    def validate_ext_item_32(self, slot_value: Any, dispatcher: CollectingDispatcher, tracker: Tracker, domain: DomainDict) -> Dict[Text, Any]:
        return self.validate_item("ext_item_32", slot_value, dispatcher, tracker, domain)

    def validate_ext_item_37(self, slot_value: Any, dispatcher: CollectingDispatcher, tracker: Tracker, domain: DomainDict) -> Dict[Text, Any]:
        return self.validate_item("ext_item_37", slot_value, dispatcher, tracker, domain)

    def validate_ext_item_52(self, slot_value: Any, dispatcher: CollectingDispatcher, tracker: Tracker, domain: DomainDict) -> Dict[Text, Any]:
        return self.validate_item("ext_item_52", slot_value, dispatcher, tracker, domain)

    def validate_off_item_8(self, slot_value: Any, dispatcher: CollectingDispatcher, tracker: Tracker, domain: DomainDict) -> Dict[Text, Any]:
        return self.validate_item("off_item_8", slot_value, dispatcher, tracker, domain)

    def validate_off_item_13(self, slot_value: Any, dispatcher: CollectingDispatcher, tracker: Tracker, domain: DomainDict) -> Dict[Text, Any]:
        return self.validate_item("off_item_13", slot_value, dispatcher, tracker, domain)
    
    def validate_off_item_23(self, slot_value: Any, dispatcher: CollectingDispatcher, tracker: Tracker, domain: DomainDict) -> Dict[Text, Any]:
        return self.validate_item("off_item_23", slot_value, dispatcher, tracker, domain)

    def validate_off_item_43(self, slot_value: Any, dispatcher: CollectingDispatcher, tracker: Tracker, domain: DomainDict) -> Dict[Text, Any]:
        return self.validate_item("off_item_43", slot_value, dispatcher, tracker, domain)

    def validate_off_item_48(self, slot_value: Any, dispatcher: CollectingDispatcher, tracker: Tracker, domain: DomainDict) -> Dict[Text, Any]:
        return self.validate_item("off_item_48", slot_value, dispatcher, tracker, domain)

    def validate_off_item_58(self, slot_value: Any, dispatcher: CollectingDispatcher, tracker: Tracker, domain: DomainDict) -> Dict[Text, Any]:
        return self.validate_item("off_item_58", slot_value, dispatcher, tracker, domain)

    def validate_ver_item_9(self, slot_value: Any, dispatcher: CollectingDispatcher, tracker: Tracker, domain: DomainDict) -> Dict[Text, Any]:
        return self.validate_item("ver_item_9", slot_value, dispatcher, tracker, domain)

    def validate_ver_item_14(self, slot_value: Any, dispatcher: CollectingDispatcher, tracker: Tracker, domain: DomainDict) -> Dict[Text, Any]:
        return self.validate_item("ver_item_14", slot_value, dispatcher, tracker, domain)

    def validate_ver_item_24(self, slot_value: Any, dispatcher: CollectingDispatcher, tracker: Tracker, domain: DomainDict) -> Dict[Text, Any]:
        return self.validate_item("ver_item_24", slot_value, dispatcher, tracker, domain)

    def validate_ver_item_39(self, slot_value: Any, dispatcher: CollectingDispatcher, tracker: Tracker, domain: DomainDict) -> Dict[Text, Any]:
        return self.validate_item("ver_item_39", slot_value, dispatcher, tracker, domain)

    def validate_ver_item_49(self, slot_value: Any, dispatcher: CollectingDispatcher, tracker: Tracker, domain: DomainDict) -> Dict[Text, Any]:
        return self.validate_item("ver_item_49", slot_value, dispatcher, tracker, domain)

    def validate_ver_item_59(self, slot_value: Any, dispatcher: CollectingDispatcher, tracker: Tracker, domain: DomainDict) -> Dict[Text, Any]:
        return self.validate_item("ver_item_59", slot_value, dispatcher, tracker, domain)

    def validate_gew_item_5(self, slot_value: Any, dispatcher: CollectingDispatcher, tracker: Tracker, domain: DomainDict) -> Dict[Text, Any]:
        return self.validate_item("gew_item_5", slot_value, dispatcher, tracker, domain)

    def validate_gew_item_10(self, slot_value: Any, dispatcher: CollectingDispatcher, tracker: Tracker, domain: DomainDict) -> Dict[Text, Any]:
        return self.validate_item("gew_item_10", slot_value, dispatcher, tracker, domain)

    def validate_gew_item_20(self, slot_value: Any, dispatcher: CollectingDispatcher, tracker: Tracker, domain: DomainDict) -> Dict[Text, Any]:
        return self.validate_item("gew_item_20", slot_value, dispatcher, tracker, domain)

    def validate_gew_item_40(self, slot_value: Any, dispatcher: CollectingDispatcher, tracker: Tracker, domain: DomainDict) -> Dict[Text, Any]:
        return self.validate_item("gew_item_40", slot_value, dispatcher, tracker, domain)
    
    def validate_gew_item_50(self, slot_value: Any, dispatcher: CollectingDispatcher, tracker: Tracker, domain: DomainDict) -> Dict[Text, Any]:
        return self.validate_item("gew_item_50", slot_value, dispatcher, tracker, domain)
    
    def validate_gew_item_55(self, slot_value: Any, dispatcher: CollectingDispatcher, tracker: Tracker, domain: DomainDict) -> Dict[Text, Any]:
        return self.validate_item("gew_item_55", slot_value, dispatcher, tracker, domain)
    

class ActionSubmitTest(Action):
    def name(self):
        return "action_submit_test"
    
    def Recode(self, item):
        code_map = {"5": "1", "4": "2", "2": "4", "1": "5"}
        return code_map.get(item, item)
    
    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict]:
        
        # These items have to be inverted (umgepolt) according to the NEO-FFI manual
        # since these items have a negated statement
        off_item_8 = self.Recode(tracker.get_slot("off_item_8"))
        ver_item_9 = self.Recode(tracker.get_slot("ver_item_9"))
        ver_item_14 = self.Recode(tracker.get_slot("ver_item_14"))
        off_item_23 = self.Recode(tracker.get_slot("off_item_23"))
        ver_item_24 = self.Recode(tracker.get_slot("ver_item_24"))
        ver_item_39 = self.Recode(tracker.get_slot("ver_item_39"))
        off_item_48 = self.Recode(tracker.get_slot("off_item_48"))
        gew_item_55 = self.Recode(tracker.get_slot("gew_item_55"))
        ver_item_59 = self.Recode(tracker.get_slot("ver_item_59"))
        
        neuro_values = [tracker.get_slot("neu_item_6"),tracker.get_slot("neu_item_21"),tracker.get_slot("neu_item_11"),
                        tracker.get_slot("neu_item_26"),tracker.get_slot("neu_item_41"),tracker.get_slot("neu_item_51")]

        extra_values = [tracker.get_slot("ext_item_2"),tracker.get_slot("ext_item_7"),tracker.get_slot("ext_item_22"),
                        tracker.get_slot("ext_item_32"),tracker.get_slot("ext_item_37"),tracker.get_slot("ext_item_52")]

        off_values = [off_item_8,tracker.get_slot("off_item_13"),off_item_23,tracker.get_slot("off_item_43"),off_item_48,
                      tracker.get_slot("off_item_58")]

        ver_values = [ver_item_9, ver_item_14, ver_item_24,ver_item_39,tracker.get_slot("ver_item_49"),ver_item_59]

        gew_values = [tracker.get_slot("gew_item_5"),tracker.get_slot("gew_item_10"),tracker.get_slot("gew_item_20"),
                        tracker.get_slot("gew_item_40"),tracker.get_slot("gew_item_50"),gew_item_55]

        # Concatenate/join lists
        neoffi_items = neuro_values + extra_values + off_values + ver_values + gew_values
        # Cast every item to integer
        neoffi_items = list(map(int, neoffi_items))
        # Compute neo ffi scores
        neo_ffi = neoffi.ComputeNeoFFI(neuro_values, extra_values, off_values, ver_values, gew_values)
        neo_ffi_result = neo_ffi.compute()
        print(neo_ffi_result)

        neuro_percent = round(neo_ffi_result[0])
        extra_percent = round(neo_ffi_result[1])
        off_percent = round(neo_ffi_result[2])
        ver_percent = round(neo_ffi_result[3])
        gew_percent = round(neo_ffi_result[4])

        dispatcher.utter_message("Danke! Hier ist deine Testauswertung:")
        dispatcher.utter_message("Neurotizismus: " + str(neuro_percent) + " %")
        dispatcher.utter_message("Extraversion: " + str(extra_percent) + " %")
        dispatcher.utter_message("Offenheit für Erfahrung: " + str(off_percent) + " %")
        dispatcher.utter_message("Verträglichkeit: " + str(ver_percent) + " %")
        dispatcher.utter_message("Gewissenhaftigkeit: " + str(gew_percent) + " %")

        # Create spiderchart of neo-ffi-results and store them as base64
        # categories = ["Neurotizismus", "Extraversion", "Offenheit für Erfahrung", "Verträglichkeit", "Gewissenhaftigkeit"]
        # spiderChart = spiderchart.SpiderChart(categories, neo_ffi_result)
        # base64spiderchart = spiderChart.plot()

        # base64pdf = CreatePDF.CreatePDF.create(base64spiderchart, neuro_percent, extra_percent, off_percent, ver_percent, gew_percent)

        # dispatcher.utter_message(image="data:image/png;base64, " + base64spiderchart)

        # chatbot_experience = [tracker.get_slot("chatbot_experience")]

        # pdf_data = {
        #     "payload": "pdf_attachment",
        #     "title": "NEO-FFI Ergebnis",
        #     "url": 'data:application/octet-stream;base64,' + base64pdf
        # }
        # dispatcher.utter_message(response="utter_explain_pdf")
        # dispatcher.utter_message(response="utter_inform_survey_finished")
        # dispatcher.utter_message(text="Download Deines Ergebnisses als PDF", json_message=pdf_data)

        dispatcher.utter_message(text="Möchtest du nun etwas plaudern?")

        # Set the test results as slots
        return [
            SlotSet("neuroticism_test_result", neuro_percent),
            SlotSet("extraversion_test_result", extra_percent),
            SlotSet("openness_test_result", off_percent),
            SlotSet("agreeableness_test_result", ver_percent),
            SlotSet("conscientiousness_test_result", gew_percent),
        ]