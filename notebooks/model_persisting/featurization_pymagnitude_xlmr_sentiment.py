from nltk import word_tokenize, sent_tokenize
import emoji
import re
import nltk
nltk.download('punkt')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
#from germansentiment import SentimentModel
import textstat
import numpy as np
import multiprocessing
import pandas as pd
import os
from tqdm import tqdm_notebook
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from numpy import hstack
from scipy import sparse
import string
from sklearn.preprocessing import Normalizer, StandardScaler, RobustScaler, MinMaxScaler, MaxAbsScaler
from pymagnitude import *
from nltk.corpus import stopwords
from typing import List

""" Library Classes and Functions """
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from typing import List
import torch
import re
from transformers import AutoModelForSequenceClassification
from transformers import TFAutoModelForSequenceClassification
from transformers import AutoTokenizer, AutoConfig
from scipy.special import softmax
import pickle
import time
import joblib

def get_glove_vectors(df, glove):
    vectors = []
    for text in tqdm_notebook(df.text.values):
        vectors.append(np.average(glove.query(word_tokenize(text)), axis = 0))
    return np.array(vectors)

def tfidf_w2v(df, idf_dict, glove):
    vectors = []
    for text in tqdm_notebook(df.text.values):
        w2v_vectors = glove.query(word_tokenize(text))
        weights = [idf_dict.get(word, 1) for word in word_tokenize(text)]
        vectors.append(np.average(w2v_vectors, axis = 0, weights = weights))
    return np.array(vectors)
    

def _preprocess(text):
    new_text = []
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)

def sentiment_analysis(df):
    # Preprocess text (username and link placeholders)
    #print("Starting XLM-Roberta sentiment analysis")
    MODEL = f"cardiffnlp/twitter-xlm-roberta-base-sentiment"

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    config = AutoConfig.from_pretrained(MODEL)

    # PT
    model = AutoModelForSequenceClassification.from_pretrained(MODEL)
    model.save_pretrained(MODEL)
    tokenizer.save_pretrained(MODEL)

    sentiments = []
    for text in df.text.values:
        text = _preprocess(text)
        encoded_input = tokenizer(text, return_tensors='pt')
        output = model(**encoded_input)
        scores = output[0][0].detach().numpy()
        scores = softmax(scores)
        sentiments.append(scores)
    
    # [neg, neu, pos]
    return np.array(sentiments)



def count_emojis(s):
    cnt = 0
    for word in word_tokenize(s):
        if emoji.is_emoji(word):
            cnt += 1

    return cnt


def emoji_count(df):
    emoticons_re = [
        '(\:\w+\:|\<[\/\\]?3|[\(\)\\\D|\*\$][\-\^]?[\:\;\=]|[\:\;\=B8][\-\^]?[3DOPp\@\$\*\\\)\(\/\|])(?=\s|[\!\.\?]|$)']
    is_emote = []

    for text in df.text.values:
        no_of_phrases = 0
        for re_patten in emoticons_re:
            no_of_phrases += len(re.findall(re_patten, text))

        no_of_phrases += count_emojis(text)

        is_emote.append(no_of_phrases)
    return np.array(is_emote).reshape(-1, 1)


def count_punctuations(df):
    puncts = []
    punctuations = set(string.punctuation)
    count = lambda l1,l2: sum([1 for x in l1 if x in l2])
    for text in df.text.values:
        puncts.append(count(text,punctuations))
    return np.array(puncts).reshape(-1,1)

def num_dots(df):
    num_dots = []
    for text in df.text.values:
        num_dots.append(text.count('.'))
    return np.array(num_dots).reshape(-1,1)


def text_features(df):
    longest_word_length = []
    mean_word_length = []
    length_in_chars = []

    for text in df.text.values:
        length_in_chars.append(len(text))
        longest_word_length.append(len(max(text.split(), key=len)))
        mean_word_length.append(np.mean([len(word) for word in text.split()]))

    longest_word_length = np.array(longest_word_length).reshape(-1, 1)
    mean_word_length = np.array(mean_word_length).reshape(-1, 1)
    length_in_chars = np.array(length_in_chars).reshape(-1, 1)

    return np.concatenate([longest_word_length, mean_word_length, length_in_chars], axis=1)

""" Main Function """

def featurization_pymagnitude_xlmr_sentiment(train_df, test_df, embedding_type):

    print('Emoji re....')
    train_emoji_re = emoji_count(train_df)
    test_emoji_re = emoji_count(test_df)

    print('Num dots....')
    train_num_dots = num_dots(train_df)
    test_num_dots = num_dots(test_df)

    print('Punctuation....')
    train_num_punctuations = count_punctuations(train_df)
    test_num_punctuations = count_punctuations(test_df)

    print('Sentiment Scores....')
    start_time = time.time()
    train_sentiment = sentiment_analysis(train_df)
    test_sentiment = sentiment_analysis(test_df)
    print("XLMR Sentiment Model took --- %s seconds ---" % (time.time() - start_time))

    print('Text Features....')
    train_text_features = text_features(train_df)
    test_text_features = text_features(test_df)

    if embedding_type == 'tfidf':
        print('TFIDF Text....')

        tfidf_word = TfidfVectorizer()

        print('TFIDF Word....')
        train_word_features = tfidf_word.fit_transform(train_df.text.values)
        test_word_features = tfidf_word.transform(test_df.text.values)

        normalizer_tfidf = MinMaxScaler()
        train_embedding_features = sparse.csr_matrix(normalizer_tfidf.fit_transform(train_word_features.todense()))

        test_embedding_features = sparse.csr_matrix(normalizer_tfidf.fit_transform(test_word_features.todense()))

    elif embedding_type == 'glove':
        print('Glove.....')
        glove = Magnitude('Joel\\vectors\\wiki.de.vec.magnitude')
        train_glove = get_glove_vectors(train_df, glove)
        test_glove = get_glove_vectors(test_df, glove)

        normalizer_glove = MinMaxScaler()
        train_glove = normalizer_glove.fit_transform(train_glove)
        test_glove = normalizer_glove.transform(test_glove)

        train_embedding_features = sparse.csr_matrix(train_glove)
        test_embedding_features = sparse.csr_matrix(test_glove)

    elif embedding_type == 'tfidf_glove':
        print('Pymagnitude.....')

        start_time = time.time()
        glove = Magnitude('../../pipeline/models/embeddings/wiki.de.vec.magnitude')
        print("Pymagnitude model load took --- %s seconds ---" % (time.time() - start_time))

        tfidf = TfidfVectorizer()
        tfidf.fit(train_df.text.values)
        idf_dict = dict(zip(tfidf.get_feature_names_out(), tfidf.idf_))
        # save the vectorizer to a file
        #with open("big5_tfidf_vectorizer.pkl", "wb") as f:
         #   pickle.dump(tfidf, f)
        
        #vectorizer_file = 'idf_vectorizer2.joblib'
        #joblib.dump(tfidf, vectorizer_file)

        start_time = time.time()
        train_glove = tfidf_w2v(train_df, idf_dict, glove)
        test_glove = tfidf_w2v(test_df, idf_dict, glove)
        print("Pymagnitude Framework took --- %s seconds ---" % (time.time() - start_time))

        normalizer_glove = MinMaxScaler()
        train_glove = normalizer_glove.fit_transform(train_glove)

        #embedding_normalizer_file = 'embedding_normalizer2.joblib'
        #joblib.dump(normalizer_glove, embedding_normalizer_file)

        test_glove = normalizer_glove.transform(test_glove)
        print("--- Normalized-features:")
        print(train_glove)

        train_embedding_features = sparse.csr_matrix(train_glove)
        test_embedding_features = sparse.csr_matrix(test_glove)

    train_features = hstack((#train_starts_with_number,
                             #train_cb_phrases,
                             train_emoji_re,
                             train_num_dots,
                             train_text_features,
                             #train_word_ratio,
                             train_sentiment,
                             #train_readability_scores,
                             train_num_punctuations))

    normalizer = MinMaxScaler()
    train_features = normalizer.fit_transform(train_features)
    
    #feature_normalizer_file = 'feature_normalizer2.joblib'
    #joblib.dump(normalizer, feature_normalizer_file)
    
    #with open("feature_normalizer.pkl", "wb") as f:
    #    pickle.dump(normalizer, f)

    train_features = sparse.csr_matrix(train_features)

    train_features = sparse.hstack((
        train_features,
        train_embedding_features
    ))

    test_features = hstack((#test_starts_with_number,
                            #test_cb_phrases,
                            test_emoji_re,
                            test_num_dots,
                            test_text_features,
                            #test_word_ratio,
                            test_sentiment,
                            #test_readability_scores,
                            test_num_punctuations))
    test_features = normalizer.transform(test_features)

    test_features = sparse.csr_matrix(test_features)
    test_features = sparse.hstack((
        test_features,
        test_embedding_features
    ))

    feature_names = [#'starts_with_number',
                     #'clickbait_phrases',
                     'train_emoji_re',
                     'num_dots',
                     'longest_word_length',
                     'mean_word_length',
                     'length_in_chars',
                     #'easy_words_ratio',
                     #'stop_words_ratio',
                     #'contractions_ratio',
                     #'hyperbolic_ratio',
                    # 'clickbait_subs_ratio',
                    # 'nonclickbait_subs_ratio',
                     'sentiment_neg',
                     'sentiment_neu',
                     'sentiment_pos',
                     #'flesch_kincaid_grade',
                    # 'dale_chall_readability_score',
                     'num_punctuations'
                     ]

    if embedding_type == 'tfidf':
        feature_names = feature_names + ['tfidf_word_' + col for col in tfidf_word.get_feature_names_out()]
    else:

        feature_names = feature_names + ['fasttext_' + str(col) for col in range(300)] # Embedding model is 300 dimensional
    print('DONE!')
    #print(train_features[:5])
    #print(train_features[-5:])
    return train_features, test_features, feature_names