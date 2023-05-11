from typing import Dict, List


import openai
import requests

from rasa.engine.graph import GraphComponent, ExecutionContext
from typing import Any, List, Optional, Text, Dict, Tuple, Union, Type
from rasa.engine.recipes.default_recipe import DefaultV1Recipe
from rasa.core.featurizers.tracker_featurizers import TrackerFeaturizer
from rasa.core.channels.channel import UserMessage
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.core.policies.policy import PolicyPrediction, Policy, SupportedData
from rasa.core.featurizers.precomputation import MessageContainerForCoreFeaturization
from rasa.shared.core.generator import TrackerWithCachedStates
from rasa_sdk import Action, Tracker
from rasa.shared.core.trackers import DialogueStateTracker
from rasa.shared.core.domain import Domain
from rasa.core.constants import (
    DIALOGUE,
    POLICY_MAX_HISTORY,
    DEFAULT_MAX_HISTORY,
    DEFAULT_POLICY_PRIORITY,
    POLICY_PRIORITY,
)

@DefaultV1Recipe.register(
    [DefaultV1Recipe.ComponentType.POLICY_WITHOUT_END_TO_END_SUPPORT], is_trainable=False
)
class ResponsePolicy(Policy):


    @classmethod
    def create(
            cls,
            config: Dict[Text, Any],
            model_storage: ModelStorage,
            resource: Resource,
            execution_context: ExecutionContext,
    ) -> Policy:
        # TODO: Implement this
        ...

    async def train(
        self,
        training_trackers: List[DialogueStateTracker],
        domain: Domain,
        **kwargs
    ) -> Resource:
        pass

    @staticmethod
    def get_default_config() -> Dict[Text, Any]:
        """Returns the default config (see parent class for full docstring)."""
        # please make sure to update the docs when changing a default parameter
        return {
            POLICY_PRIORITY: 10.0,
            POLICY_MAX_HISTORY: 10,
        }

    async def predict_action_probabilities(
        self,
        tracker: DialogueStateTracker,
        domain: Domain,
        rule_only_data: Optional[Dict[Text, Any]] = None,
    ) -> List[float]:
        print(tracker.latest_message)
        #intent = tracker.latest_message.get("intent", {}).get("name")
        intent = getattr(tracker.latest_message,"intent")
        #intent = "ask_for_help"
        entities = getattr(tracker.latest_message,"entities")

        # Prepare the context and the class label
        context = getattr(tracker.latest_message,"text")
        class_label = f"{intent}_{'_'.join([e['entity'] for e in entities])}"

        print("Found entities:" + str(entities))
        print("Classlabel: " + str(class_label))
        # Use the openai library to fine-tune the GPT-3 model and generate a response
        # response = openai.Completion.create(
        #     #engine=self.model,
        #     engine="text-davinci-003",
        #     prompt=f"{context}\n\n{class_label}:",
        #     max_tokens=1024,
        #     n=1,
        #     stop=None,
        #     temperature=0.5,
        # ).choices[0].text

        response = openai.Completion.create(
            model="text-davinci-003",
            prompt=f"{context}\n\n{class_label}:",
            max_tokens=1024,
            n=1,
            stop=None,
            temperature=0
        )
        #print(res["choices"][0]["text"])

        # Return the generated response as the next action
        return [
            {
                "action": "utter_gpt3_response",
                "confidence": 1.0,
                "response": response
            }
        ]