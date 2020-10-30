#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 17:00:50 2020

@author: jkreula
"""
import pandas as pd
import re
import os
from typing import List
import nltk
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
import spacy
import en_core_web_sm

# Load core English language model
nlp = en_core_web_sm.load()

def clean_text(text: str) -> str:
    '''
    Clean HTML markup, remove non-words and put text to lowercase. 

    Parameters
    ----------
    text : str
        Text to be cleaned.

    Returns
    -------
    str
        Cleaned text.

    '''
    text = re.sub('<[^>]*>','',text)
    text = re.sub(r'[\W]+',' ',text.lower())
    return text.strip()

def get_tokens(text: str) -> List[str]:
    '''
    Split cleaned text into a list of individual words.

    Parameters
    ----------
    text : str
        Text to be split.

    Returns
    -------
    List[str]
        Split text, i.e. a list of words.

    '''
    # Define common stop words
    stop_words = stopwords.words('english')
    tokens=nltk.word_tokenize(text)
    return [word for word in tokens if word not in stop_words]

def get_stemmed_tokens(text: str) -> List[str]:
    '''
    Split cleaned text into a list of individual words with Porter stemming.

    Parameters
    ----------
    text : str
        Text to be split.

    Returns
    -------
    List[str]
        Split text, i.e. a list of words, in their root form.

    '''
    stop_words = stopwords.words('english')
    stemmer = SnowballStemmer('english')
    return [stemmer.stem(word) for word in text.split() if word not in stop_words]

def get_lemmatized_tokens(text: str) -> List[str]:
    stop_words = stopwords.words('english')
    return [word.lemma_ for word in nlp(text) if word.text not in stop_words]


def clean_and_tokenise(text: str) -> List[str]:
    '''
    Clean text and tokenise using clean_text() and get_tokens() functions.

    Parameters
    ----------
    text : str
        Text to be cleaned and tokenised.

    Returns
    -------
    List[str]
        List of cleaned tokens.

    '''
    return get_tokens(clean_text(text))

def clean_and_tokenise_with_stemming(text: str) -> List[str]:
    '''
    Clean text and tokenise with stemming using clean_text() and 
    get_stemmed_tokens() functions.

    Parameters
    ----------
    text : str
        Text to be cleaned and tokenised with stemming.

    Returns
    -------
    List[str]
        List of cleaned tokens with stemming.

    '''
    
    return get_stemmed_tokens(clean_text(text))


def clean_and_tokenise_with_lemmatization(text: str) -> List[str]:
    return get_lemmatized_tokens(clean_text(text))


if __name__ == "__main__":
    
    test = "<h1> :-( </a> A :) 71 test testing <div> tester :( review ! :-(!"
    test_ = clean_text(test)
    print(test_)
    test__ = clean_and_tokenise(test)
    print(test__)
    
    test_lem = clean_and_tokenise_with_lemmatization(test)
    print(test_lem)
    
    test_stem = clean_and_tokenise_with_stemming(test)
    print(test_stem)
    
    # Current working directory
    CWD = os.getcwd()
    
    # Path to data
    data_folder = "Data"
    DATA_PATH = os.path.join(CWD,data_folder)
    
    data_filename="imdb_data.csv"
    df = pd.read_csv(os.path.join(DATA_PATH,data_filename))
    df_test = df.copy()
    df['Review_text'] = df['Review_text'].apply(clean_text)    