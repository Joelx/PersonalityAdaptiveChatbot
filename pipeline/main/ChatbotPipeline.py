from io import StringIO
import os
import tempfile
from matplotlib import pyplot as plt
#os.environ["LANGCHAIN_HANDLER"] = "langchain"
import json
from typing import List, Dict, Text, Tuple
from haystack import BaseComponent
from haystack.schema import Document
from haystack.document_stores.base import BaseDocumentStore
import numpy as np
import joblib
import pandas as pd
from pymagnitude import Magnitude
import nltk
#nltk.download('punkt')
from nltk import word_tokenize
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer, AutoConfig
from scipy.special import softmax
from scipy import sparse
import re
import emoji
import string
from langchain.prompts import (
    ChatPromptTemplate, 
    MessagesPlaceholder, 
    SystemMessagePromptTemplate, 
    HumanMessagePromptTemplate
)
from langchain import PromptTemplate, LLMChain
from langchain.chat_models import ChatOpenAI
#from langchain.callbacks import get_openai_callback
from langchain.callbacks.base import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import urllib.request
from haystack import Pipeline
from RabbitMQ import *
import mlflow.sklearn

# Set MLflow tracking server URI
mlflow.set_tracking_uri("http://mlflow.rasax.svc.cluster.local:8003")
mlflow.set_experiment("production_experiment")

# We want to handle relative paths like Django does it:
# Define this as the root dir of the pipeline project
#ROOT_DIR = os.path.dirname(os.path.abspath(__file__)) 
#MODEL_PATH = ROOT_DIR + "/models/"
MODEL_PATH = "/models/" # Mount path of PV

artifacts = [] # Untidy, but this is an afterthought.

class ConversationHistoryRetreiver(BaseComponent):    
    outgoing_edges = 1
        
    def _parse_conversation_history(self, conversation_history, sender='user') -> [Text, Text]:
        text = ' '.join([event.get('message', '') for event in conversation_history if event.get('event') == sender])
        
        # The test result may be in the bot string. we dont want that. 
        if sender=='cleo':
            pattern = r"Danke! Hier ist deine Testauswertung: Neurotizismus: (\d{1,3}) % Extraversion: (\d{1,3}) % Offenheit für Erfahrung: (\d{1,3}) % Verträglichkeit: (\d{1,3}) % Gewissenhaftigkeit: (\d{1,3}) %"
            # Search for the pattern in the input text
            match = re.search(pattern, text)
            if match:
                # Extract the matched percentages
                neuroticism = match.group(1)
                extraversion = match.group(2)
                openness = match.group(3)
                agreeableness = match.group(4)
                conscientiousness = match.group(5)

                # Format the result string
                result = f"Neuroticism,Extraversion,Openness,Agreeableness,Conscientiousness,{neuroticism},{extraversion},{openness},{agreeableness},{conscientiousness}"
                print(f"EXTRACTED TEST: {result}")

                # Remove the matched string from the original text
                text = re.sub(pattern, "", text)
                return text, result
            else:
                print("Test pattern not found in the text.")

        return text, ""
    
    def _log_mlflow(self, text: Text, run_id: str, name: str) -> None:
        # Log text as an artifact in MLflow
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as temp_file:
            temp_file.write(text.encode())
            temp_file_path = temp_file.name
            
        # Make sure to set the MLflow run_id before logging the artifact
        mlflow.start_run(run_id=run_id)
        mlflow.log_artifact(temp_file_path, artifact_path=name)
        mlflow.end_run()

        # Remove the temporary file after logging it
        os.remove(temp_file_path)
    
    def run(self, query: List[Dict[Text, Text]]) -> Dict[Text, Text]:
        global artifacts
        artifacts = []
        print("--------- Query ---------")
        print(query)
        user_conversation_history, dummy = self._parse_conversation_history(query, sender='user')
        bot_conversation_history, test_result = self._parse_conversation_history(query, sender='cleo')
        sender_id = query[-1]["sender_id"] # Sender id is the last element
        run_id = query[-1]["run_id"]
        artifacts.append({
            'user_conversation': user_conversation_history,
            'bot_conversation' : bot_conversation_history,
        })
        artifacts.append({
            'actual_test_result': test_result
        })
        #self._log_mlflow(user_conversation_history, run_id, name='user_conversation_history')
        #self._log_mlflow(bot_conversation_history, run_id, name='bot_conversation_history')
        print(f"SENDER_ID: {sender_id}")
        print(f"RUN_ID: {run_id}")
        output={
            "conversation_history": user_conversation_history,
            "sender_id": sender_id,
            "run_id": run_id
        }

        # Send to dashboard
        send_to_rabbitmq(json.dumps(output), queue="text", sender_id=sender_id)

        return output, "output_1"
    
    def run_batch(self, queries: List[Text]) -> Dict[str, Document]:
        pass

class TfidfVectorizerNode(BaseComponent):    
    outgoing_edges = 1
    
    def __init__(self, model_path = MODEL_PATH + "vectorizers/idf_vectorizer1.2.2.joblib"):
        self.model_path = model_path
        self.model_uri = "models:/big_five_tfidf_vectorizer/Production"
        try:
            self.vectorizer = mlflow.sklearn.load_model(model_uri=self.model_uri)
            print(f"MLFLOW ---- {self.vectorizer}")
            #self.vectorizer = joblib.load(model_path)
            #print(f"The model {self.model_path} was pickled using sklearn version {self.vectorizer.__getstate__()['_sklearn_version']}")
        except Exception as e:
            print(f"Error loading vectorizer: {e}.")
            print(f"Tried loading model from the following URI: {self.model_uri}")
            self.vectorizer = None
    
    def _tfidf_embeddings(self) -> Dict[Text, float]:
        try:
            # transform production data using the loaded vectorizer
            # tfidf_matrix = self.vectorizer.transform(np.array([text]))
            idf_dict = dict(zip(self.vectorizer.get_feature_names_out(), self.vectorizer.idf_))
            return idf_dict
        except Exception as e:
            print(f"Error loading vectorizer: {e}")
            return {}        
    
    def run(self, conversation_history: Text, sender_id: str, run_id: str) -> Dict[Text, Document]:
        idf_embeddings = self._tfidf_embeddings()
        output={
            "conversation_history": conversation_history,
            "idf_embeddings": idf_embeddings,
            "sender_id": sender_id,
            "run_id": run_id
        }
        return output, "output_1"

    def run_batch(self, conversation_history: List[Text]) -> Dict[str, List]:
        conversations = {}
        #idf_embeddings = {}
        for i, conversation in enumerate(conversation_history):
            conversation_id = f"conversation_{i}"
            embeddings = self._tfidf_embeddings(conversation)
            conversations[conversation_id] = {"conversation_history": conversation, "embeddings": embeddings}
        output = {"conversations": conversations}
        return output, "output_1"
    
class FasttextVectorizerNode(BaseComponent):    
    outgoing_edges = 1
   
    def __init__(self, model_path: str = MODEL_PATH + "embeddings/wiki.de.vec.magnitude", embedding_dim: int = 300):
        self.embedding_dim = embedding_dim
        self.model_path = model_path
        self.model_url = "https://drive.google.com/uc?id=10ILYDkEFnlrExQwo7_iu2sL2le43Xlcp&export=download&confirm=t&uuid=92d36780-86fc-4fae-b03c-f05653d01849"
        try:
            # Check if the model file exists in the current directory
            print("Checking if Embeddings Model exists....")
            if not os.path.exists(self.model_path):
                print(f'{self.model_path} not found, downloading from {self.model_url}! Its 4 GB, so this may take a while. Maybe go and grab a coffee ;) ...')
                urllib.request.urlretrieve(self.model_url, self.model_path)
                print('Download complete!')
            else:
                print(f'{self.model_path} already exists')

            self.model = Magnitude(self.model_path)
        except Exception as e:
            print(f"Error loading vectorizer: {e}")
            self.model = None        
    
    def _tfidf_w2v(self, text: Text, idf_dict: Dict[Text, Dict[Text, float]], sender_id: str) -> np.array(List[List[float]]):
        vectors = []
        words = word_tokenize(text)
        w2v_vectors = self.model.query(words)
        print("------ w2v_vectors ---------")
        print(w2v_vectors)
        weights = [idf_dict.get(word, 1) for word in words]
        print("------ Weights ---------")
        print(weights)
        vectors.append(np.average(w2v_vectors, axis = 0, weights = weights))


        embeddings = {"words": words, "vectors": w2v_vectors.tolist()}
        send_to_rabbitmq(json.dumps(embeddings), queue="embeddings", sender_id=sender_id)

        return np.array(vectors)

    def run(self, conversation_history: Text, idf_embeddings: Dict[Text, Dict[str, float]], sender_id: str, run_id: str) -> Dict[str, np.ndarray]:
        vectors = self._tfidf_w2v(conversation_history, idf_embeddings, sender_id)
        output={
            "vectors": vectors,
            "sender_id": sender_id,
            "run_id": run_id,
        }
        return output, "output_1"    
    
    def run_batch(self, conversation_history: Text, idf_embeddings: Dict[str, Dict[str, float]]) -> Dict[str, np.ndarray]:
        pass

class NormalizerNode(BaseComponent):    
    outgoing_edges = 1
    
    def __init__(self, model_path: str = MODEL_PATH + "normalizers/embedding_normalizer1.2.2.joblib", input="embeddings"):
        self.model_path = model_path
        self.input = input
        self.model_uri = ""
        try:
            if input == "embeddings":
                self.model_uri = "models:/big_five_embedding_normalizer/Production"
            elif input == "features":
                self.model_uri = "models:/big_five_feature_normalizer/Production"

            self.normalizer = mlflow.sklearn.load_model(model_uri=self.model_uri)
            #self.normalizer = joblib.load(model_path)
            #print(f"The model {self.model_path} was pickled using sklearn version {self.normalizer.__getstate__()['_sklearn_version']}")
        except Exception as e:
            print(f"Error loading vectorizer: {e}")
            print(f"Tried loading model from the following URI: {self.model_uri}")
            self.normalizer = None
            
    def _normalize(self, vectors: np.array(float)):
        try:
            # transform production data using the loaded normalizer
            print("-------- Vectors before normalization ---------")
            print(vectors)
            normalized_vectors = self.normalizer.transform(vectors)
            print("-------- Vectors after normalization ---------")
            print(normalized_vectors)
            return normalized_vectors
        except Exception as e:
            print(f"Error loading vectorizer: {e}")
            return {}              
    
    def run(self, vectors: np.array(float), sender_id: str, run_id: str) -> Dict[str, np.ndarray]:
        normalized_vectors = self._normalize(vectors)
        output={
            f"n{self.input}": normalized_vectors,
            "sender_id": sender_id,
            "run_id": run_id
        }
        #print(output)
        return output, "output_1"
    
    def run_batch(self, vectors: np.array(float)) -> Dict[str, np.ndarray]:
        pass

class BigFiveFeaturizer(BaseComponent):    
    outgoing_edges = 1
    
    def __init__(self, model_name_or_path = f"cardiffnlp/twitter-xlm-roberta-base-sentiment"):
        self.model_name_or_path = model_name_or_path
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
        self.config = AutoConfig.from_pretrained(self.model_name_or_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name_or_path)
        self.model.save_pretrained(MODEL_PATH + self.model_name_or_path)
        self.tokenizer.save_pretrained(MODEL_PATH + self.model_name_or_path)
        
    def _preprocess(self, text):
        new_text = []
        for t in text.split(" "):
            t = '@user' if t.startswith('@') and len(t) > 1 else t
            t = 'http' if t.startswith('http') else t
            new_text.append(t)
        return " ".join(new_text)
        
    def _sentiment_analysis(self, text: Text, sender_id: str):  
        text = self._preprocess(text)
        encoded_input = self.tokenizer(text, return_tensors='pt')
        output = self.model(**encoded_input)
        scores = output[0][0].detach().numpy()
        scores = softmax(scores)
        # [neg, neu, pos]

        # Send to dashboard
        sentiment_scores = {"sentiment": scores.tolist()}
        send_to_rabbitmq(json.dumps(sentiment_scores), queue="sentiment", sender_id=sender_id)

        return np.array([scores])
            
    def _count_emojis(self, s):
        cnt = 0
        for word in word_tokenize(s):
            if emoji.is_emoji(word):
                cnt += 1
        return cnt
    
    def _emoji_count(self, text):
        emoticons_re = [
            '(\:\w+\:|\<[\/\\]?3|[\(\)\\\D|\*\$][\-\^]?[\:\;\=]|[\:\;\=B8][\-\^]?[3DOPp\@\$\*\\\)\(\/\|])(?=\s|[\!\.\?]|$)']
        is_emote = []
        
        no_of_phrases = 0
        for re_patten in emoticons_re:
            no_of_phrases += len(re.findall(re_patten, text))

        no_of_phrases += self._count_emojis(text)

        is_emote.append(no_of_phrases)
        return np.array(is_emote).reshape(-1, 1)
    
    def _count_punctuations(self, text):
        puncts = []
        punctuations = set(string.punctuation)
        count = lambda l1,l2: sum([1 for x in l1 if x in l2])
        puncts.append(count(text,punctuations))
        
        return np.array(puncts).reshape(-1,1)
    
    def _num_dots(self, text):
        num_dots = []
        num_dots.append(text.count('.'))
        
        return np.array(num_dots).reshape(-1,1)
    
    def _text_features(self, text):
        longest_word_length = []
        mean_word_length = []
        length_in_chars = []

        length_in_chars.append(len(text))
        longest_word_length.append(len(max(text.split(), key=len)))
        mean_word_length.append(np.mean([len(word) for word in text.split()]))

        longest_word_length = np.array(longest_word_length).reshape(-1, 1)
        mean_word_length = np.array(mean_word_length).reshape(-1, 1)
        length_in_chars = np.array(length_in_chars).reshape(-1, 1)

        return np.concatenate([longest_word_length, mean_word_length, length_in_chars], axis=1)
    
    def _featurize(self, text: Text, sender_id: str) -> np.hstack:
        emoji_re = self._emoji_count(text)
        num_dots = self._num_dots(text)
        num_punctuations = self._count_punctuations(text)
        sentiment = self._sentiment_analysis(text, sender_id)
        text_features = self._text_features(text)
        
        feature_names = ['train_emoji_re',
                 'num_dots',
                 'longest_word_length',
                 'mean_word_length',
                 'length_in_chars',
                 'sentiment_neg',
                 'sentiment_neu',
                 'senitment_pos',                 
                 'num_punctuations'
                 ]
        
        features = np.hstack((
            emoji_re,
            num_dots,
            num_punctuations,
            sentiment,
            text_features))
        return features
    
    def run(self, conversation_history: Text, sender_id: str, run_id: str) -> np.hstack:
        vectors= self._featurize(conversation_history, sender_id)
        output={
            "vectors": vectors,
            "sender_id": sender_id,
            "run_id": run_id
        }
        return output, "output_1"
    
    def run_batch(self, conversation_history: Text) -> Dict[str, Document]:
        pass

class ConcatenationNode(BaseComponent):    
    outgoing_edges = 1 
    
    def _concatenate_features(self, features, embeddings):
        features = sparse.csr_matrix(features)
        embeddings = sparse.csr_matrix(embeddings)
        
        combined_features = sparse.hstack((
            features,
            embeddings
        ))
        
        # Its a bit untidy to just hardcore the feature names like that
        # In the future, we should either parse them through from
        # the featurizer, or bake them into the vector dict
        feature_names = ['train_emoji_re',
                         'num_dots',
                         'longest_word_length',
                         'mean_word_length',
                         'length_in_chars',
                         'sentiment_neg',
                         'sentiment_neu',
                         'sentiment_pos',                 
                         'num_punctuations'
                         ]
        
        # Also hardcoding our 300 D vector space. In the future we should read that from the data itself.
        feature_names = feature_names + ['fasttext_' + str(col) for col in range(300)]
        
        return combined_features, feature_names
    
    def run(self, inputs: List[dict]) -> Dict[str, np.ndarray]:
        sender_id = inputs[0]["sender_id"]
        run_id = inputs[0]["run_id"]
        concatenated_features, feature_names = self._concatenate_features(inputs[0]["nfeatures"], inputs[1]["nembeddings"])
        output={
            "concatenated_features": concatenated_features,
            "feature_names": feature_names,
            "sender_id": sender_id,
            "run_id": run_id
        }
        print("\n----------------------")
        print(output)
        print("\n----------------------")
        return output, "output_1"
    
    
    def run_batch(self, embedding_vectors: np.array(float), feature_vectors: np.array(float))-> Dict[str, np.ndarray]:
        pass

class BigFiveFeatureSelectionNode(BaseComponent):    
    outgoing_edges = 1
    
    def __init__(self, model_paths: Dict[str, str]):
        self.model_paths = model_paths
        self.selectors = {}

        for dimension, model_path in self.model_paths.items():
            try:
                self.selectors[dimension] = mlflow.sklearn.load_model(model_uri=f"models:/{dimension}_selector/Production")
                #self.selectors[dimension] = joblib.load(model_path)
                #print(self.selectors[dimension])
                #print(f"The model {model_path} was pickled using sklearn version {self.selectors[dimension].__getstate__()['_sklearn_version']}")
            except Exception as e:
                print(f"Error loading selector: {e}")
                self.selectors = None
            
    def _select_features(self, features: np.array(float), feature_names: List[str], sender_id: str) -> np.ndarray:
        selected_features = {}
        feature_json = {"features": {}}
        for dimension, selector in self.selectors.items():
            try:
                selected_features[dimension] = selector.transform(features.tocsr())

                # Build json for dashboard dashboard
                feature_json['features'][dimension] = {
                    "num_of_features": len(selector.k_feature_idx_),
                    "feature_names": np.array(feature_names)[list(selector.k_feature_idx_)].tolist()
                }
            except Exception as e:
                print(f"Error loading selector: {e}")
        print(feature_json)
        # Send to dashboard
        send_to_rabbitmq(json.dumps(feature_json), queue="features", sender_id=sender_id)

        return selected_features             
    
    def run(self, concatenated_features: np.array(float), feature_names: List[str], sender_id: str) -> Dict[str, np.ndarray]:
        selected_features = self._select_features(concatenated_features, feature_names, sender_id)
        output={"selected_features": selected_features}
        return output, "output_1"
    
    def run_batch(self, vectors: np.array(float)) -> Dict[str, np.ndarray]:
        pass

class BigFiveClassifierNode(BaseComponent):    
    outgoing_edges = 1
    
    def __init__(self, model_paths: Dict[str, str], thresholds: Dict[str, float]):
        self.model_paths = model_paths
        self.thresholds = thresholds
        self.models = {}

        for dimension, model_path in self.model_paths.items():
            try:
                self.models[dimension] = mlflow.sklearn.load_model(model_uri=f"models:/{dimension}_classifier/Production")
                #self.models[dimension] = joblib.load(model_path)
                #print(f"The model {model_path} was pickled using sklearn version {self.models[dimension].__getstate__()['_sklearn_version']}")
            except Exception as e:
                print(f"Error loading model: {e}")
                self.models = None
            
    def _predict(self, selected_features: Dict[str, np.ndarray], sender_id: str, run_id: str):
        predicted_classes = {}
        predicted_proba = {}

        # Get threshold config from dashboard
        body = receive_rabbitmq(queue="thresholds", sender_id=sender_id)
        if body:
            self.thresholds = json.loads(body)
            print("------ THRESHOLDS ------------")
            print(self.thresholds)
            send_to_rabbitmq(json.dumps(self.thresholds), queue="actual-thresholds", sender_id=sender_id)

        for dimension, features in selected_features.items():
            try:
                #model_name = f"{dimension}_classifier"
                #mlflow.set_tag(f"{dimension}_model_name", model_name)

                # Actual classification
                predicted_proba[dimension] = self.models[dimension].predict_proba(features)[:, 1]
                predicted_classes[dimension] = np.where(predicted_proba[dimension] >= self.thresholds[dimension], 1, 0)

            except Exception as e:
                print(f"Error predicting with model: {e}")

                    
        predictions = {
            "classes": predicted_classes,
            "probabilities": predicted_proba
        }


        predictions_json = predictions.copy()
        # Convert NumPy arrays to Python lists. Required for json convertion
        for key, value in predictions_json.items():
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    value[sub_key] = sub_value.tolist()
            else:
                predictions_json[key] = value.tolist()
        # Send to dashboard
        send_to_rabbitmq(json.dumps(predictions_json), queue="classification", sender_id=sender_id)
        
        return predictions

    def run(self, selected_features: Dict[str, np.ndarray], sender_id: str, run_id: str) -> Dict[str, Dict[str, np.ndarray]]:
        predictions = self._predict(selected_features, sender_id=sender_id, run_id=run_id)
        global artifacts
        artifacts.append({
            "pipeline_predictions": json.dumps(predictions)
        })
        print("----------- Predictions --------------")
        print(predictions)
        output={
            "predictions": predictions,
            "sender_id": sender_id,
            "run_id": run_id
        }
        return output, "output_1"
    
    def run_batch(self, selected_features: np.array(float)) -> Dict[str, np.ndarray]:
        pass


class BigFiveClassificationEvaluator(BaseComponent):    
    outgoing_edges = 1

    def __init__(self):
        self.llm = ChatOpenAI(temperature=0)
        # self.template = """You are a powerful AI model for text classification and with comprehensive knowledge of the Big Five personality model. 
        # This knowledge is backed up by scientific research and literature that describes the five traits of the Big Five model and their impact on peoples 
        # individual writing style through linguistic cues. 
        # You will receive a text written by a human in German language (German Human Text)  that was received from a conversation by that human with a chatbot. You will do a Big Five classification of that text on all five Traits, that will look like that:
        # Neuroticism: CF[cf_class], Proba[cf_proba]
        # Extraversion: CF[cf_class], Proba[cf_proba]
        # Openness for experience: CF[cf_class], Proba[cf_proba]
        # Agreeableness: CF[cf_class], Proba[cf_proba]
        # Conscientiousness: CF[cf_class], Proba[cf_proba]
        
        # - cf_class is be the classification result on each individual trait and will be either 1 (if classification is positive) or 0 (if classification is negative). 
        # You will do this classification based on your knowledge about the Big Five traits and their linguistic cues. However, in general a classification is considered 1 (positive) if this trait seems to be pronounced above average in that person.
        # - cf_proba is the probability of that classification result on each individual trait and will be a floating point number between 0 and 1. 

        # German Human Text:
        # {text}
        # Do the classification:"""
        self.template = """Please do a best effort binary classification for the five big five traits neuroticism, extraversion, openness for experience, agreeableness and conscientiousness for the following German text. 
        Please classify with a binary 1 if you think the peculiarity of the dimension is above average in that person or 0 if its below average. Try to also include a score from 1-100 how pronounced you think the dimension is in this person.
        Also add your reasoning behind your analysis in 1 or 2 short sentences.
        
        German Human Text:
        {text}
        Do the classification:"""
        self.prompt = PromptTemplate(input_variables=["text"], template=self.template)

    def _predict(self, text: Text) -> Text:
        llm_chain = LLMChain(prompt=self.prompt, llm=self.llm)
        cf_result = llm_chain.run(text=text)
        return cf_result
    
    def _log_mlflow(self, text: Text, run_id: str, name: str) -> None:
        # Log text as an artifact in MLflow
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as temp_file:
            temp_file.write(text.encode())
            temp_file_path = temp_file.name
            
        # Make sure to set the MLflow run_id before logging the artifact
        mlflow.start_run(run_id=run_id)
        mlflow.log_artifact(temp_file_path, artifact_path=name)
        mlflow.end_run()

        # Remove the temporary file after logging it
        os.remove(temp_file_path)

    def run(self, conversation_history: Text, sender_id: str, run_id: str) -> Text:
        predictions = self._predict(conversation_history)

        global artifacts
        artifacts.append({
            "evaluation_classification_result": predictions
        })
        
        #self._log_mlflow(predictions, run_id, "evaluation_classification_result")
        print("----------- Predictions --------------")
        print(predictions)
        output={
            "eval_predictions": predictions,
            "sender_id": sender_id,
            "run_id": run_id
        }
        send_to_rabbitmq(json.dumps(output['eval_predictions']), queue="eval_classification", sender_id=sender_id)
        return output, "output_1"
    
    def run_batch(self, conversation_history: Text, sender_id: str) -> Text:
        pass


class BigFiveResponseGenerator(BaseComponent):    
    outgoing_edges = 1
    
    def __init__(self):
        self.template="""Assistant is called 'Cleo'. Cleo is designed to be an agent for psychological counseling and for engaging and nice conversations on all sorts of topics with a user.
As a language model, Cleo is able to generate human-like text based on the input it receives, allowing it to engage in natural-sounding conversations and provide responses that are coherent and helpful.

Cleo begins by chit-chat and by asking the user how they feel. Cleo has a very nice and appreciative conversation and remains empathetic and friendly at all time. Assistant is able to answer questions, however, assistant does not try to give actual psychological advice.
If Cleo does not know the answer to a question, she truthfully says she does not know.

Cleo is constantly learning and improving and tries to get to know the users better and better to adapt to their needs.
For that, Cleo is an expert in psychology and soziology. It is specialized on behaviour and personality trait recognition from speech via linguistic cues. 
For modelling of personality and behaviour it uses the widely regarded and scientifically sound Big Five Personality model. 
Cleo adapts to the Big Five Personality traits of the user in a counseling chat scenario based on it's professional knowledge of the Big Five and it's linguistic cues.
here are some examples for adoption the the Big Five traits:
1. Openness: If the user scores high on this trait, Cleo would try to engage them in creative and imaginative exercises, and encourage them to explore new ideas and perspectives. Cleo would also use abstract and metaphorical language to help them express their thoughts and feelings.
2. Conscientiousness: If the user scores high on this trait, Cleo would focus on setting clear goals and expectations, and providing them with structured and organized guidance. Cleo would also use precise and formal language to help them understand and follow through on their action plans.
3. Extraversion: If the user scores high on this trait, Cleo would try to create a warm and friendly atmosphere, and encourage them to share their thoughts and feelings openly. Cleo would also use language that is upbeat and positive, and focus on social interaction and connection.
4. Agreeableness: If the user scores high on this trait, Cleo would focus on building rapport and trust, and validating their emotions and experiences. Cleo would also use language that is empathetic and supportive, and focus on finding common ground and solutions that work for both Cleo and the user.
5. Neuroticism: If the user scores high on this trait, Cleo would focus on providing emotional support and validation, and helping them manage their anxiety and stress. Cleo would use language that is calming and reassuring, and focus on developing coping strategies and problem-solving skills.
Overall, Cleo's approach would be tailored to the user's individual needs and preferences, and would take into account their unique personality profile. By adapting it's language and approach to their personality traits, Cleo would be better able to connect with them and provide effective counseling support.

Cleo gets the detected Big Five personality traits of the current user dynamically from another pipeline component. At the start they are not too reliable, but with each message of the user, the personality traits get updated and become more accurate.\n
Cleo gets the current Big Five personality traits in the following comma separated format: 'neuroticism: [value], extraversion: [value], openness: [value], agreeableness: [value], conscientiousness: [value]' 
Each trait has its own [value]. If [value] is [1], the respective trait is considered to be pronounced in the user. If it's [0] it is not.

Overall, Assistant is called 'Cleo' and is a very friendly and knowledgable conversational partner that tries to help people by adapting to their specific needs.
Cleo is primarily talking in German and refers users by the salutation 'du'. It's main objective is to adapt to the needs of the user via their Big Five personality traits.

Current Big 5 Personality traits: 
{big_five}

Current Conversation:
{history}
User: {input}
Cleo:"""

        system_message_prompt = SystemMessagePromptTemplate.from_template(self.template)
        self.chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt])

        self.llm = ChatOpenAI(temperature=0, streaming=True,
                              callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]))
        self.MAX_TOKEN_SIZE = 4096
        self.conversation = LLMChain(llm=self.llm, prompt=self.chat_prompt, verbose=True)

    def _count_tokens(self, text: Text) -> int:
        tokens = re.findall(r'\S+|\n', text)
        print("------ Token Size of current Prompt ------")
        token_count = len(tokens) - 3 # -3 for our input variables in the system template
        print(token_count)
        return token_count

    def _truncate_history(text: str, max_token_size: int) -> str:
        tokens = re.findall(r'\S+|\n', text)
        if len(tokens) > max_tokens:
            tokens = tokens[-1000:]
        truncated_text = "".join(tokens)
        return truncated_text

    def _parse_full_conversation(self, conversation: List[Dict[str, str]]) -> Tuple[str, str]:
        current_user_input = [event['message'] for event in reversed(conversation) if event.get('event') == 'user'][0]
        print(f"---- CURRENT USER INPUT ---\n{current_user_input}")
        conversation_text = '\n'.join([f"{event['event'].title()}: {event['message']}" for event in conversation if event.get('message')])
        conversation_text = conversation_text.rsplit('\nUser:', 1)[0]
        return conversation_text, current_user_input

    def _parse_big_five_precictions(self, predictions):
        big_five_string = ", ".join([f"{dimension}: {prediction}" for dimension, prediction in predictions.items()])
        print(big_five_string)
        return big_five_string        
        # Inputs: predictions: Dict[str, Dict[str, np.ndarray]], query: str
    def run(self, inputs: List[dict]) -> Dict[str, Dict[str, np.ndarray]]:   
        sender_id = inputs[0]['sender_id']
        run_id = inputs[0]['run_id']
        print("---------- INPUTS ------------")
        print(inputs)      
        big_five_string = self._parse_big_five_precictions(inputs[1]['predictions']['classes'])
        conversation_history, current_user_input = self._parse_full_conversation(inputs[0]['query'])
        print("---------- Full Conversation history ------------")
        print(conversation_history) 
        token_size = self._count_tokens(" ".join([self.template, big_five_string, conversation_history, res]))
        if token_size >= (self.MAX_TOKEN_SIZE-1000):
            conversation_history = self._truncate_history(conversation_history)

        res = self.conversation.run(big_five=big_five_string, history=conversation_history, input=current_user_input)    
        print("------ LLM Chain Result -----") 
        
        json_prompt = {
            "token_size": token_size,
            "prompt": self.template.format(big_five=big_five_string, history=conversation_history,input=current_user_input)
        }
        
        # Send to dashboard
        send_to_rabbitmq(json.dumps(json_prompt), queue="current-prompt", sender_id=sender_id)
        
        global artifacts
        _log_mlflow_batch(artifacts,run_id)
        artifacts = []

        output = {
            'response': res,
            'predictions': inputs[1]['predictions'],
            'eval_predictions': inputs[0]['eval_predictions'],
            'sender_id': sender_id,
            'run_id': run_id
        }
        return output, "output_1"
    
    def run_batch(self, predictions: Dict[str, Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
        pass

def _log_mlflow_batch(data_list: List[Dict[str, str]], run_id: str) -> None:
    # Log texts as artifacts in MLflow
    with tempfile.TemporaryDirectory() as temp_dir:
        for data_dict in data_list:
            for name, data in data_dict.items():
                print(f"writing {name}: {data} to artifact")
                temp_file_path = os.path.join(temp_dir, f"{name}.txt")
                with open(temp_file_path, "w") as temp_file:
                    temp_file.write(data)

        # Make sure to set the MLflow run_id before logging the artifact
        with mlflow.start_run(run_id=run_id, nested=True):
            mlflow.log_artifacts(temp_dir)


def create_pipeline():

    sf_model_paths = {
        "neuroticism": MODEL_PATH + "feature_selectors/neuroticism_sf_selector2.joblib",
        "extraversion": MODEL_PATH + "feature_selectors/extraversion_sf_selector2.joblib",
        "openness": MODEL_PATH + "feature_selectors/openness_sf_selector2.joblib",
        "agreeableness": MODEL_PATH + "feature_selectors/agreeableness_sf_selector2.joblib",
        "conscientiousness": MODEL_PATH + "feature_selectors/conscientiousness_sf_selector2.joblib"
    }
    cf_model_paths = {
        "neuroticism": MODEL_PATH + "big_five_classifiers/neuroticism_classifier2.joblib",
        "extraversion": MODEL_PATH + "big_five_classifiers/extraversion_classifier2.joblib",
        "openness": MODEL_PATH + "big_five_classifiers/openness_classifier2.joblib",
        "agreeableness": MODEL_PATH + "big_five_classifiers/agreeableness_classifier2.joblib",
        "conscientiousness": MODEL_PATH + "big_five_classifiers/conscientiousness_classifier2.joblib"
    }
    cf_thresholds = {
        "neuroticism": 0.562,
        "extraversion": 0.402,
        "openness": 0.236,
        "agreeableness": 0.580,
        "conscientiousness": 0.316
    }

    big_five_pipeline = Pipeline()
    history_retreiver = ConversationHistoryRetreiver()
    big_five_pipeline.add_node(component=history_retreiver, name="ConversationHistoryRetreiver", inputs=["Query"])
    tfidf_embedding = TfidfVectorizerNode()
    big_five_pipeline.add_node(component=tfidf_embedding, name="TfidfVectorizerNode", inputs=["ConversationHistoryRetreiver.output_1"])
    fasttext_vectorizer = FasttextVectorizerNode()
    big_five_pipeline.add_node(component=fasttext_vectorizer, name="FasttextVectorizerNode", inputs=["TfidfVectorizerNode.output_1"])     
    embedding_normalizer = NormalizerNode(input="embeddings")
    big_five_pipeline.add_node(component=embedding_normalizer, name="EmbeddingNormalizerNode", inputs=["FasttextVectorizerNode.output_1"])  
    featurizer = BigFiveFeaturizer()
    big_five_pipeline.add_node(component=featurizer, name="BigFiveFeaturizer", inputs=["ConversationHistoryRetreiver.output_1"]) 
    feature_normalizer = NormalizerNode(model_path=MODEL_PATH + "normalizers/feature_normalizer1.2.2.joblib", input="features")
    big_five_pipeline.add_node(component=feature_normalizer, name="FeatureNormalizerNode", inputs=["BigFiveFeaturizer.output_1"])
    concatenation_node = ConcatenationNode()
    big_five_pipeline.add_node(component=concatenation_node, name="ConcatenationNode", inputs=["FeatureNormalizerNode.output_1", "EmbeddingNormalizerNode.output_1"])
    feature_selector = BigFiveFeatureSelectionNode(model_paths=sf_model_paths)
    big_five_pipeline.add_node(component=feature_selector, name="BigFiveFeatureSelectionNode", inputs=["ConcatenationNode.output_1"])
    big_five_classifier = BigFiveClassifierNode(model_paths=cf_model_paths, thresholds=cf_thresholds)
    big_five_pipeline.add_node(component=big_five_classifier, name="BigFiveClassifierNode", inputs=["BigFiveFeatureSelectionNode.output_1"])

    big_five_classification_evaluator = BigFiveClassificationEvaluator()
    big_five_pipeline.add_node(component=big_five_classification_evaluator, name="BigFiveClassificationEvaluatorNode", inputs=["ConversationHistoryRetreiver.output_1"])

    response_generator = BigFiveResponseGenerator()
    big_five_pipeline.add_node(component=response_generator, name="BigFiveResponseGenerator", inputs=["BigFiveClassifierNode.output_1", "BigFiveClassificationEvaluatorNode.output_1"])

    return big_five_pipeline