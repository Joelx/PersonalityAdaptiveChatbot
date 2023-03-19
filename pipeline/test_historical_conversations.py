#!/usr/bin/env python3

# AS SEEN IN: https://gist.github.com/alwx/b426b7b573ff963c85c65ea6466528d7

"""Script to run `rasa test` on historic conversations.
This script extracts conversations from a Rasa Open Source tracker store or
Rasa X API and runs `rasa test core` on these conversations.
The script is intended to be run on conversations that were:
1. Originally processed by an assistant not trained with `UnexpecTEDIntentPolicy`
   included the pipeline.
2. Originally processed by an assistant trained on the same training data which is used to
   train the model passed to this script.
For any other set of conversations that were originally processed by an assistant trained
with `UnexpecTEDIntentPolicy` included in the pipeline, you don't need to run this script
and you can filter the interesting conversations by checking if `action_unlikely_intent`
was predicted in the original conversation itself.
    Show help message:
    $ python test_historical_conversations.py -h
    Typical usage example (with tracker stores):
    $ python test_historical_conversations.py \
        --filename my_logs.yml --out my_logs \
        --minimum-timestamp 1626163200
    A usage example for cases when conversations need to be fetched from Rasa X API:
    $ python test_historical_conversations.py \
        --rasa-x-api --filename my_logs.yml --out my_logs \
        --model intent_ted_model.tar.gz \
        --minimum-timestamp 1626163200
The tolerance parameter of UnexpecTEDIntentPolicy can also be swept to help
you find a suitable value without needing to retrain the policy ensemble
multiple times. To run a sweep add `--minimum-tolerance` and/or
`--maximum-tolerance` arguments to your command like so:
    $ python test_historical_conversations.py \n
        --filename my_logs.yml --out my_logs \n
        --minimum-tolerance 0.0 --maximum-tolerance 0.5
Results will appear in directories labeled like so:
      my_logs/tol0.00
      my_logs/tol0.05
      ...
      my_logs/tol0.50
You can adjust the step size of the sweep by setting the
`--tolerance-step` argument. Note that the minimum step size is 0.05.
"""
import argparse
import asyncio
import os
import pathlib
import tarfile
import shutil
import tempfile
import itertools
from aiohttp import ClientSession
from urllib import parse
import pickle
import numpy as np
from typing import List, Text, Any, Dict, Optional, Union
import time
import datetime

import rasa.cli.test
import rasa.core.utils
from rasa.core.tracker_store import TrackerStore

import rasa.shared.utils.cli
import rasa.shared.utils.io
from rasa.shared.constants import (
    DEFAULT_MODELS_PATH,
    DEFAULT_ENDPOINTS_PATH,
)
from rasa.shared.core import events
from rasa.shared.core.training_data.structures import Story
from rasa.shared.core.training_data.story_writer.yaml_story_writer import (
    YAMLStoryWriter,
)
from rasa.shared.core.events import Event, UserUttered
from rasa.shared.core.domain import Domain
from rasa.utils.tensorflow.constants import TOLERANCE

USERNAME = os.environ.get("RASA_X_USERNAME", "admin")
PASSWORD = os.environ.get("RASA_X_PASSWORD", "rasa")
RASA_X_URL = os.environ.get("RASA_X_URL", "http://localhost:5002")

yaml_writer = YAMLStoryWriter()


def _create_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        "test_historical_conversations.py",
        description="Check for problems with the structure of the Alembic migration "
        "tree when upgrading across Rasa X versions. Exits with non-zero status if at "
        "least one problem was found.",
    )

    parser.add_argument(
        "--successes",
        action="store_true",
        default=False,
        help="If set successful predictions will be written to a file.",
    )
    parser.add_argument(
        "--no-errors",
        action="store_true",
        default=False,
        help="If set incorrect predictions will NOT be written to a file.",
    )
    parser.add_argument(
        "--no-warnings",
        action="store_true",
        default=False,
        help="If set prediction warnings will NOT be written to a file.",
    )
    parser.add_argument(
        "--rasa-x-api",
        action="store_true",
        default=False,
        help="If conversations need to be fetched from Rasa X API",
    )
    parser.add_argument(
        "--evaluate-model-directory",
        default=False,
        action="store_true",
        help="Should be set to evaluate models trained via "
        "'rasa train core --config <config-1> <config-2>'. "
        "All models in the provided directory are evaluated "
        "and compared against each other.",
    )
    parser.add_argument(
        "-m",
        "--model",
        nargs="+",
        default=DEFAULT_MODELS_PATH,
        help="Path to a pre-trained model. If it is a 'tar.gz' file that model file "
        "will be used. If it is a directory, the latest model in that directory "
        "will be used (exception: '--evaluate-model-directory' flag is set). "
        "If multiple 'tar.gz' files are provided, all those models will be compared.",
    )
    parser.add_argument(
        "--filename",
        type=str,
        default="tests/historical_stories.yml",
        help="A filename for historical test stories extracted from Rasa X.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="rasa_x_results",
        help="Output path for any files created during the test evaluation of historical test stories.",
    )
    parser.add_argument(
        "--endpoints",
        type=str,
        default=DEFAULT_ENDPOINTS_PATH,
        help="Endpoint configuration file specifying the tracker store "
        "and event broker.",
    )
    parser.add_argument(
        "--minimum-date",
        type=str,
        help=(
            "Minimum date of events to be exported in dd/mm/YYYY format. The constraint is applied "
            "in a 'greater than or equal' comparison."
        ),
    )
    parser.add_argument(
        "--maximum-date",
        type=str,
        help=(
            "Maximum date of events to be exported in dd/mm/YYYY format. The constraint is "
            "applied in a 'less than' comparison."
        ),
    )
    parser.add_argument(
        "--minimum-tolerance",
        type=float,
        help=(
            "Minimum tolerance value when sweeping the tolerance parameter. "
            "Must be greater than or equal to 0.0 and less than 1.0."
        ),
    )
    parser.add_argument(
        "--maximum-tolerance",
        type=float,
        help=(
            "Maximum tolerance value when sweeping the tolerance parameter. "
            "Must be greater than or equal to 0.0 and less than or equal to "
            "1.0."
        ),
    )
    parser.add_argument(
        "--tolerance-step",
        default=0.05,
        type=float,
        help=(
            "Step size when sweeping the tolerance parameter. "
            "Must be greater than or equal to 0.05 and less than 1.0."
        ),
    )

    return parser


def _arg_date_to_timestamp(date: Optional[Text]) -> Optional[float]:
    if date is not None:
        return time.mktime(datetime.datetime.strptime(date, "%d/%m/%Y").timetuple())
    return None


def _assert_max_timestamp_is_greater_than_min_timestamp(
    minimum_timestamp: Optional[float] = None,
    maximum_timestamp: Optional[float] = None,
) -> None:
    """Inspect CLI timestamp parameters.
    Prints an error and exits if a maximum timestamp is provided that is smaller
    than the provided minimum timestamp.
    Args:
        minimum_timestamp: Minimum timestamp of events that are published.
            If `None`, apply no such constraint.
        maximum_timestamp: Maximum timestamp of events that are published.
            If `None`, apply no such constraint.
    """
    if (
        minimum_timestamp is not None
        and maximum_timestamp is not None
        and maximum_timestamp < minimum_timestamp
    ):
        rasa.shared.utils.cli.print_error_and_exit(
            f"Maximum timestamp '{maximum_timestamp}' is smaller than minimum "
            f"timestamp '{minimum_timestamp}'. Exiting."
        )


def _is_conversation_within_time_range(
    sorted_events: List[Event],
    minimum_timestamp: Optional[float] = None,
    maximum_timestamp: Optional[float] = None,
) -> bool:
    """Checks if the time when the conversation has started is within the
    specified time range.
    Args:
        sorted_events: List of serialized sorted events.
        minimum_timestamp: Minimum timestamp. If `None`, apply no such constraint.
        maximum_timestamp: Maximum timestamp.  If `None`, apply no such constraint.
    Returns:
        `True` if the conversation has started within the specified time range,
        `False` otherwise.
    """
    if (
        sorted_events
        and (
            minimum_timestamp is None or sorted_events[0].timestamp >= minimum_timestamp
        )
        and (
            maximum_timestamp is None or sorted_events[0].timestamp < maximum_timestamp
        )
    ):
        return True

    return False


async def _login(username: Text, password: Text, url: Text) -> Text:
    """Log into Rasa X.
    Args:
        username: Username.
        password: Password of the user.
        url: URL of the Rasa X instance which should logged into.
    Returns:
        The JWT access token of the user.
    """
    url = parse.urljoin(url, "api/auth")
    payload = {"username": username, "password": password}
    async with ClientSession() as session:
        response = await session.post(url, json=payload)
        assert response.status == 200

        response_body = await response.json()
        access_token = response_body["access_token"]
        assert access_token
        return access_token


def _client_session(access_token: Text) -> ClientSession:
    headers = {"Authorization": f"Bearer {access_token}"}
    return ClientSession(headers=headers)


async def _fetch_conversations(session: ClientSession) -> List[Dict[Text, Any]]:
    """Gets conversations from Rasa X API.
    Args:
        session: An initialized client session.
    Returns:
        A list of all conversations.
    """
    url = parse.urljoin(RASA_X_URL, "api/conversations")
    response = await session.get(url)
    if response.status != 200:
        rasa.shared.utils.cli.print_error(f"Unable to call GET {url}.")
        return []
    return await response.json()


async def _fetch_full_conversation(
    session: ClientSession, conversation_id: Text
) -> Optional[Dict[Text, Any]]:
    """
    Gets a full conversation from Rasa X API.
    Args:
        session: An initialized client session.
        conversation_id: An ID of the conversation to fetch.
    Returns:
        A full conversation for a specified `conversation_id`.
    """
    url = parse.urljoin(RASA_X_URL, "api/conversations/" + conversation_id)
    response = await session.get(url)
    if response.status != 200:
        rasa.shared.utils.cli.print_warning(f"Unable to call GET {url}.")
        return None
    return await response.json()


def story_to_yaml(story: Story, conversation_id: Text) -> Text:
    """
    Transform a story to YAML.
    Args:
        story: A story.
        conversation_id: An ID of the conversation that was fetched.
    Returns:
        A YAML containing all the story steps.
    """
    name = f"Story from Conversation ID {conversation_id}"
    for step in story.story_steps:
        step.block_name = name

    return yaml_writer.dumps(story.story_steps, is_test_story=True, is_appendable=True)


def story_events_contains_valid_intents(events: List[Event], domain: Domain) -> bool:
    """Checks whether events contain intents from the supplied domain.
    Args:
        events: Conversation events to check.
        domain: Loaded domain of the model supplied in arguments.
    Returns:
        Whether events contain valid intents.
    """
    for event in events:
        if isinstance(event, UserUttered):
            if event.intent.get("name") not in domain.intents:
                return False
    return True


async def fetch_api_yaml_stories(
    model_domain: Domain,
    minimum_timestamp: Optional[float] = None,
    maximum_timestamp: Optional[float] = None,
) -> Text:
    """Fetch stories from running Rasa X instance.
    Args:
        model_domain: Loaded domain of the model supplied in arguments.
        minimum_timestamp: Minimum timestamp of events that are published.
            If `None`, apply no such constraint.
        maximum_timestamp: Maximum timestamp of events that are published.
            If `None`, apply no such constraint.
    Returns:
        Extracted stories in YAML format.
    """
    access_token = await _login(USERNAME, PASSWORD, RASA_X_URL)
    yaml_data = ""
    num_conversations = 0
    async with _client_session(access_token) as session:
        conversations = await _fetch_conversations(session)
        for conversation in conversations:
            if conversation["n_user_messages"] == 0:
                continue
            conversation_id = conversation["sender_id"]
            full_conversation = await _fetch_full_conversation(session, conversation_id)
            if full_conversation:
                story_events = events.deserialise_events(full_conversation["events"])
                sorted_events = list(sorted(story_events, key=lambda e: e.timestamp))
                if _is_conversation_within_time_range(
                    sorted_events, minimum_timestamp, maximum_timestamp
                ) and story_events_contains_valid_intents(sorted_events, model_domain):
                    story = Story.from_events(story_events)
                    yaml_data += story_to_yaml(story, conversation_id)
                    num_conversations += 1

    print(f"Number of conversations fetched: {num_conversations}")
    return f'version: "2.0"\nstories:\n{yaml_data}\n'


def fetch_tracker_store_yaml_stories(
    model_domain: Domain,
    minimum_timestamp: Optional[float] = None,
    maximum_timestamp: Optional[float] = None,
) -> Text:
    """Fetch stories from a tracker store.
    Args:
        model_domain: Loaded domain of the model supplied in arguments.
        minimum_timestamp: Minimum timestamp of events that are published.
            If `None`, apply no such constraint.
        maximum_timestamp: Maximum timestamp of events that are published.
            If `None`, apply no such constraint.
    Returns:
        Extracted stories in YAML format.
    """
    endpoints = rasa.core.utils.read_endpoints_from_path(args.endpoints)
    if not endpoints.tracker_store:
        rasa.shared.utils.cli.print_error_and_exit(
            f"Could not find a `tracker_store` section in the supplied endpoints file."
        )

    yaml_data = ""
    num_conversations = 0
    tracker_store = TrackerStore.create(endpoints.tracker_store)
    for sender_id in tracker_store.keys():
        tracker = tracker_store.retrieve(sender_id)
        sorted_events = list(sorted(tracker.events, key=lambda e: e.timestamp))
        if _is_conversation_within_time_range(
            sorted_events, minimum_timestamp, maximum_timestamp
        ) and story_events_contains_valid_intents(sorted_events, model_domain):
            story = Story.from_events(sorted_events)
            yaml_data += story_to_yaml(story, sender_id)
            num_conversations += 1

    print(f"Number of conversations fetched: {num_conversations}")
    return f'version: "2.0"\nstories:\n{yaml_data}\n'


def get_model_tolerance(model_path: Union[Text, List[Text]]) -> float:
    if isinstance(model_path, list):
        if len(model_path) > 1:
            raise ValueError("Cannot perform tolerance sweep on multiple models.")
        model_path = model_path[0]
    model_path = pathlib.Path(rasa.model.get_local_model(model_path))

    with tarfile.open(model_path) as tar:
        for member in tar.getmembers():
            if member.name.endswith("unexpected_intent_policy.meta.pkl"):
                f = tar.extractfile(member)
                metadata = pickle.load(f)
                return metadata[TOLERANCE]

    raise ValueError(f"Model at {model_path} does not have an UnexpecTEDIntentPolicy.")


def get_model_domain(model_path: Union[Text, List[Text]], out_path: Text) -> Domain:
    if isinstance(model_path, list):
        if len(model_path) > 1:
            raise ValueError("Cannot perform tolerance sweep on multiple models.")
        model_path = model_path[0]
    model_path = pathlib.Path(rasa.model.get_local_model(model_path))

    with tarfile.open(model_path) as tar:
        for member in tar.getmembers():
            if member.name == "core/domain.yml":
                tar.extract(member, f"{out_path}/domain.yml")
                domain = Domain.load(f"{out_path}/domain.yml")
                return domain
        raise ValueError(f"Domain file not found in the model tar file {model_path}. Please check your trained model.")


def set_model_tolerance(model_path: Union[Text, List[Text]], tolerance: float,) -> None:

    if isinstance(model_path, list):
        if len(model_path) > 1:
            raise ValueError("Cannot perform tolerance sweep on multiple models.")
        model_path = model_path[0]
    model_path = pathlib.Path(rasa.model.get_local_model(model_path))
    unpacked_model = rasa.model.unpack_model(model_path)
    core_path = pathlib.Path(unpacked_model, "core")

    for uip_path in core_path.glob("policy_*_UnexpecTEDIntentPolicy"):

        meta_path = uip_path / "unexpected_intent_policy.meta.pkl"

        with open(meta_path, "rb") as f:
            metadata = pickle.load(f)
            metadata[TOLERANCE] = tolerance
        with open(meta_path, "wb") as f:
            pickle.dump(metadata, f)

    tmp_dir = tempfile.mkdtemp()
    tmp_tar_path = pathlib.Path(tmp_dir, model_path.name)

    with tarfile.open(tmp_tar_path, "w:gz") as tar:
        for elem in os.scandir(unpacked_model):
            tar.add(elem.path, arcname=elem.name)

    shutil.rmtree(unpacked_model)
    shutil.move(tmp_tar_path, model_path)
    shutil.rmtree(tmp_dir)


def get_tolerances(
    minimum_tolerance: Optional[float],
    maximum_tolerance: Optional[float],
    tolerance_step: Optional[float],
) -> List[float]:
    if minimum_tolerance is None and maximum_tolerance is None:
        return []

    if minimum_tolerance is None:
        minimum_tolerance = 0.0
    elif minimum_tolerance < 0.0:
        rasa.shared.utils.cli.print_warning(
            "minimum_tolerance cannot be less than 0. Setting to 0.0."
        )
        minimum_tolerance = 0.0
    if maximum_tolerance is None:
        maximum_tolerance = 1.0
    elif maximum_tolerance > 1.0:
        rasa.shared.utils.cli.print_warning(
            "maximum_tolerance cannot be greater than 1. Setting to 1.0."
        )
        maximum_tolerance = 1.0

    if minimum_tolerance >= maximum_tolerance:
        raise ValueError(
            "'minimum_tolerance' must be strictly less than 'maximum_tolerance'."
        )

    if tolerance_step <= 0.0:
        raise ValueError("tolerance_step must be greater than 0.")
    if tolerance_step < 0.05:
        rasa.shared.utils.cli.print_warning(
            "The minimum tolerance_step is 0.05. Setting to 0.005."
        )
        tolerance_step = 0.05

    return list(np.arange(minimum_tolerance, maximum_tolerance + 1E-5, tolerance_step))


if __name__ == "__main__":
    parser = _create_argument_parser()
    args = parser.parse_args()

    minimum_timestamp = _arg_date_to_timestamp(args.minimum_date)
    maximum_timestamp = _arg_date_to_timestamp(args.maximum_date)
    _assert_max_timestamp_is_greater_than_min_timestamp(
        minimum_timestamp, maximum_timestamp
    )

    model_domain = get_model_domain(args.model, args.out)

    loop = asyncio.get_event_loop()
    if args.rasa_x_api:
        loop = asyncio.get_event_loop()
        yaml = loop.run_until_complete(
            fetch_api_yaml_stories(model_domain, minimum_timestamp, maximum_timestamp)
        )
    else:
        yaml = fetch_tracker_store_yaml_stories(model_domain, minimum_timestamp, maximum_timestamp)

    rasa.shared.utils.io.write_text_file(yaml, args.filename)

    args.stories = args.filename
    args.e2e = False

    tolerances = get_tolerances(
        args.minimum_tolerance, args.maximum_tolerance, args.tolerance_step
    )
    # If a tolerance range is specified run sweep otherwise just evaluate as is.
    if tolerances:

        pretty_tols = "[" + ", ".join([f"{t:0.2f}" for t in tolerances]) + "]"
        print(f"Sweeping tolerance across the following values: {pretty_tols}.")
        out = args.out
        orig_tol = get_model_tolerance(args.model)

        for tol in tolerances:
            set_model_tolerance(args.model, tol)
            args.out = pathlib.Path(out, f"tol{tol:0.2f}")
            rasa.cli.test.run_core_test(args)
        set_model_tolerance(args.model, orig_tol)

    else:
        rasa.cli.test.run_core_test(args)