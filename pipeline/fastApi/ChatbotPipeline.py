from typing import List, Dict, Text, Tuple
from haystack import BaseComponent
from haystack.schema import Document
from haystack.document_stores.base import BaseDocumentStore
import numpy as np
import joblib
from pymagnitude import Magnitude
import nltk
nltk.download('punkt')
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
import os
import urllib.request
from haystack import Pipeline
#import config

# We want to handle relative paths like Django does it:
# Define this as the root dir of the pipeline project
ROOT_DIR = os.path.dirname(os.path.abspath(__file__)) 
MODEL_PATH = ROOT_DIR + "/models/"

class ConversationHistoryRetreiver(BaseComponent):    
    outgoing_edges = 1
        
    def _parse_user_conversation_history(self, conversation_history) -> Text:
        user_text = ' '.join([event.get('message', '') for event in conversation_history if event.get('event') == 'user'])
        return user_text
    
    def run(self, query: List[Dict[Text, Text]]) -> Dict[Text, Text]:
        print("--------- Query ---------")
        print(query)
        user_conversation_history = self._parse_user_conversation_history(query)
        output={"conversation_history": user_conversation_history}
        return output, "output_1"
    
    def run_batch(self, queries: List[Text]) -> Dict[str, Document]:
        pass

class TfidfVectorizerNode(BaseComponent):    
    outgoing_edges = 1
    
    def __init__(self, model_path = MODEL_PATH + "vectorizers/idf_vectorizer1.2.2.joblib"):
        self.model_path = model_path
        try:
            self.vectorizer = joblib.load(model_path)
            #print(f"The model {self.model_path} was pickled using sklearn version {self.vectorizer.__getstate__()['_sklearn_version']}")
        except Exception as e:
            print(f"Error loading vectorizer: {e}")
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
    
    def run(self, conversation_history: Text) -> Dict[Text, Document]:
        idf_embeddings = self._tfidf_embeddings()
        output={
            "conversation_history": conversation_history,
            "idf_embeddings": idf_embeddings
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
    
    def _tfidf_w2v(self, text: Text, idf_dict: Dict[Text, Dict[Text, float]]) -> np.array(List[List[float]]):
        vectors = []
        w2v_vectors = self.model.query(word_tokenize(text))
        weights = [idf_dict.get(word, 1) for word in word_tokenize(text)]
        vectors.append(np.average(w2v_vectors, axis = 0, weights = weights))
        return np.array(vectors)

    def run(self, conversation_history: Text, idf_embeddings: Dict[Text, Dict[str, float]]) -> Dict[str, np.ndarray]:
        vectors = self._tfidf_w2v(conversation_history, idf_embeddings)
        output={"vectors": vectors}
        return output, "output_1"    
    
    def run_batch(self, conversation_history: Text, idf_embeddings: Dict[str, Dict[str, float]]) -> Dict[str, np.ndarray]:
        pass

class NormalizerNode(BaseComponent):    
    outgoing_edges = 1
    
    def __init__(self, model_path: str = MODEL_PATH + "normalizers/embedding_normalizer1.2.2.joblib", input="embeddings"):
        self.model_path = model_path
        self.input = input
        try:
            self.normalizer = joblib.load(model_path)
            #print(f"The model {self.model_path} was pickled using sklearn version {self.normalizer.__getstate__()['_sklearn_version']}")
        except Exception as e:
            print(f"Error loading vectorizer: {e}")
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
    
    def run(self, vectors: np.array(float)) -> Dict[str, np.ndarray]:
        normalized_vectors = self._normalize(vectors)
        output={f"n{self.input}": normalized_vectors}
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
        
    def _sentiment_analysis(self, text: Text):  
        text = self._preprocess(text)
        encoded_input = self.tokenizer(text, return_tensors='pt')
        output = self.model(**encoded_input)
        scores = output[0][0].detach().numpy()
        scores = softmax(scores)
        # [neg, neu, pos]
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
        
        no_of_phenvires = 0
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
    
    def _featurize(self, text) -> np.hstack:
        emoji_re = self._emoji_count(text)
        num_dots = self._num_dots(text)
        num_punctuations = self._count_punctuations(text)
        sentiment = self._sentiment_analysis(text)
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
    
    def run(self, conversation_history: Text) -> np.hstack:
        vectors= self._featurize(conversation_history)
        output={"vectors": vectors}
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
        concatenated_features, feature_names = self._concatenate_features(inputs[0]["nfeatures"], inputs[1]["nembeddings"])
        output={
            "concatenated_features": concatenated_features,
            "feature_names": feature_names   
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
                self.selectors[dimension] = joblib.load(model_path)
                #print(self.selectors[dimension])
                #print(f"The model {model_path} was pickled using sklearn version {self.selectors[dimension].__getstate__()['_sklearn_version']}")
            except Exception as e:
                print(f"Error loading selector: {e}")
                self.selectors = None
            
    def _select_features(self, features: np.array(float)):
        selected_features = {}
        for dimension, selector in self.selectors.items():
            try:
                selected_features[dimension] = selector.transform(features.tocsr())
            except Exception as e:
                print(f"Error loading selector: {e}")
        return selected_features             
    
    def run(self, concatenated_features: np.array(float)) -> Dict[str, np.ndarray]:
        selected_features = self._select_features(concatenated_features)
        print("----------- Selected Features --------------")
        print(selected_features)
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
                self.models[dimension] = joblib.load(model_path)
                #print(f"The model {model_path} was pickled using sklearn version {self.models[dimension].__getstate__()['_sklearn_version']}")
            except Exception as e:
                print(f"Error loading model: {e}")
                self.models = None
            
    def _predict(self, selected_features: Dict[str, np.ndarray]):
        predicted_classes = {}
        predicted_proba = {}
        for dimension, features in selected_features.items():
            try:
                #predicted_classes[dimension] = self.models[dimension].predict(features)
                predicted_proba[dimension] = self.models[dimension].predict_proba(features)[:,1]
                predicted_classes[dimension] = np.where(predicted_proba[dimension] >= self.thresholds[dimension], 1, 0)
            except Exception as e:
                print(f"Error predicting with model: {e}")
                
        predictions = {
            "classes": predicted_classes,
            "probabilities": predicted_proba
        }
        
        return predictions

    def run(self, selected_features: Dict[str, np.ndarray]) -> Dict[str, Dict[str, np.ndarray]]:
        predictions = self._predict(selected_features)
        print("----------- Predictions --------------")
        print(predictions)
        output={"predictions": predictions}
        return output, "output_1"
    
    def run_batch(self, selected_features: np.array(float)) -> Dict[str, np.ndarray]:
        pass

class BigFiveResponseGenerator(BaseComponent):    
    outgoing_edges = 1
    
    def __init__(self):
        self.template="""Assistant is called 'Cleo'. Cleo is designed to be an agent for psychological counseling and for engaging and nice conversations on all sorts of topics with a user.
As a language model, Cleo is able to generate human-like text based on the input it receives, allowing it to engage in natural-sounding conversations and provide responses that are coherent and helpful.

Cleo begins by chit-chat and by asking the user how they feel. Cleo has a very nice and appreciative conversation and remains empathetic and friendly at all time. Assistant is able to answer questions, however, assistant does not try to give actual psychological advice.
If Cleo does not know the answer to a question, it truthfully says it does not know.

Cleo is constantly learning and improving and tries to get to know the users better and better to adapt to their needs.
For that, Cleo is an expert in psychology and soziology. It is specialized on behaviour and personality trait recognition from speech via linguistic cues. 
For modelling of personality and behaviour it uses the widely regarded and scientifically sound Big Five Personality model. 
Cleo adapts to the Big Five Personality traits of the user in a counseling chat scenario based on it's professional knowledge of the Big Five and it's linguistic cues.
Cleo gets the detected Big Five personality traits of the current user dynamically from another pipeline component. At the start they are not too reliable, but with each message of the user, the personality traits get updated and become more accurate. The score on each dimension ranges from 1 to 100, with 1 representing the minimum and 100 the maximum score on the respective Big Five personality trait.\n
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

    def _parse_full_conversation(self, conversation: List[Dict[str, str]]) -> Tuple[str, str]:
        current_user_input = [event['message'] for event in reversed(conversation) if event.get('event') == 'user'][0]
        conversation_text = '\n'.join([f"{event['event'].title()}: {event['message']}" for event in conversation])
        conversation_text = conversation_text.rsplit('\nUser:', 1)[0]
        return conversation_text, current_user_input

    def _parse_big_five_precictions(self, predictions):
        big_five_string = ", ".join([f"{dimension}: {prediction}" for dimension, prediction in predictions.items()])
        print(big_five_string)
        return big_five_string        
        # Inputs: predictions: Dict[str, Dict[str, np.ndarray]], query: str
    def run(self, inputs: List[dict]) -> Dict[str, Dict[str, np.ndarray]]:   
        print("---------- INPUTS ------------")
        print(inputs)      
        big_five_string = self._parse_big_five_precictions(inputs[1]['predictions']['classes'])
        conversation_history, current_user_input = self._parse_full_conversation(inputs[0]['query'])
        print("---------- Full Conversation history ------------")
        print(conversation_history) 
        res = self.conversation.run(big_five=big_five_string, history=conversation_history, input=current_user_input)    
        self._count_tokens(" ".join([self.template, big_five_string, conversation_history, res]))
        print("------ LLM Chain Result -----") 
        print(res)  
        
        output = {'response': res}
        return output, "output_1"
    
    def run_batch(self, predictions: Dict[str, Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
        pass

def create_pipeline():
    #os.environ["OPENAI_API_KEY"] = config.OPENAI_API_KEY

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
        "neuroticism": 0.578,
        "extraversion": 0.478,
        "openness": 0.178,
        "agreeableness": 0.494,
        "conscientiousness": 0.299
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
    response_generator = BigFiveResponseGenerator()
    big_five_pipeline.add_node(component=response_generator, name="BigFiveResponseGenerator", inputs=["BigFiveClassifierNode.output_1", "Query"])

    return big_five_pipeline