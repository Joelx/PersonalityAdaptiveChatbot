from __future__ import annotations
import zlib

import base64
import json
import logging

from tqdm import tqdm
from typing import Optional, Any, Dict, List, Text, Tuple, Union
from pathlib import Path

import rasa.utils.io
import rasa.shared.utils.io
from rasa.engine.graph import ExecutionContext
from rasa.engine.recipes.default_recipe import DefaultV1Recipe
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.shared.core.domain import State, Domain
from rasa.shared.core.events import ActionExecuted
from rasa.core.featurizers.tracker_featurizers import TrackerFeaturizer
from rasa.core.featurizers.tracker_featurizers import MaxHistoryTrackerFeaturizer
from rasa.core.featurizers.tracker_featurizers import FEATURIZER_FILE
from rasa.shared.exceptions import FileIOException
from rasa.core.policies.policy import PolicyPrediction, Policy, SupportedData
from rasa.shared.core.trackers import DialogueStateTracker
from rasa.shared.core.generator import TrackerWithCachedStates
from rasa.shared.utils.io import is_logging_disabled
from rasa.core.constants import (
    MEMOIZATION_POLICY_PRIORITY,
    DEFAULT_MAX_HISTORY,
    POLICY_MAX_HISTORY,
    POLICY_PRIORITY,
)
from rasa.shared.core.constants import ACTION_LISTEN_NAME

logger = logging.getLogger(__name__)


@DefaultV1Recipe.register(
    DefaultV1Recipe.ComponentType.POLICY_WITHOUT_END_TO_END_SUPPORT, is_trainable=False
)
class PolicyTest(Policy):

    @staticmethod
    def get_default_config() -> Dict[Text, Any]:
        """Returns the default config (see parent class for full docstring)."""
        # please make sure to update the docs when changing a default parameter
        return {
            "enable_feature_string_compression": True,
            "use_nlu_confidence_as_score": False,
            POLICY_PRIORITY: MEMOIZATION_POLICY_PRIORITY,
            POLICY_MAX_HISTORY: DEFAULT_MAX_HISTORY,
        }

    def _standard_featurizer(self) -> MaxHistoryTrackerFeaturizer:
        # Memoization policy always uses MaxHistoryTrackerFeaturizer
        # without state_featurizer
        return MaxHistoryTrackerFeaturizer(
            state_featurizer=None, max_history=self.config[POLICY_MAX_HISTORY]
        )

    def __init__(
        self,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
        featurizer: Optional[TrackerFeaturizer] = None,
        lookup: Optional[Dict] = None,
    ) -> None:
        """Initialize the policy."""
        super().__init__(config, model_storage, resource, execution_context, featurizer)



    def train(
        self,
        training_trackers: List[TrackerWithCachedStates],
        domain: Domain,
        **kwargs: Any,
    ) -> Resource:
       ...

    def _prediction_result(
        self, action_name: Text, tracker: DialogueStateTracker, domain: Domain
    ) -> List[float]:
        result = self._default_predictions(domain)
        if action_name:
            if (
                self.config["use_nlu_confidence_as_score"]
                and tracker.latest_message is not None
            ):
                # the memoization will use the confidence of NLU on the latest
                # user message to set the confidence of the action
                score = tracker.latest_message.intent.get("confidence", 1.0)
            else:
                score = 1.0

            result[domain.index_for_action(action_name)] = score

        return result

    def predict_action_probabilities(
        self,
        tracker: DialogueStateTracker,
        domain: Domain,
        rule_only_data: Optional[Dict[Text, Any]] = None,
        **kwargs: Any,
    ) -> List[Dict[Text, Union[Text, float]]]:
        import openai
        openai.api_key = "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"

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


        res = openai.Completion.create(
            model="text-davinci-003",
            prompt=f"{context}\nClass: {class_label}\nResponse: ",
            max_tokens=1024,
            n=1,
            stop=None,
            temperature=0
        )

        response = res["choices"][0]["text"]
        print("Response: " + str(response))
        # Return the generated response as the next action
        #return [{"name": "utter_gpt3_response", "confidence": 1.0}], [{"custom_message": response}]
        return [{"name": "utter_gpt3_response", "confidence": 1.0}], response
        # return PolicyPrediction ([
        #     {
        #         "policy_name": "PolicyTest",
        #         "action": "utter_gpt3_response",
        #         "confidence": 1.0,
        #         "response": response
        #     }
        # ])



    def persist(self) -> None:
        """Persists the policy to storage."""
    ...

    @classmethod
    def load(
        cls,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
        **kwargs: Any,
    ) -> PolicyTest:
        """Loads a trained policy (see parent class for full docstring)."""
        ...

