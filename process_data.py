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
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

def clean_text(text: str) -> str:
    '''
    Clean HTML markup, find emoticons, remove non-words and put text to lowercase. 
    Found emoticons are concatenated at the end of the cleaned text.

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
    emoticons = re.findall(r'(?::|;|=)(?:-)?(?:\)|\(|D|P)',text)
    text = re.sub(r'[\W]+',' ',text.lower()) + ' '.join(emoticons).replace('-','')
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
    return [word for word in text.split() if word not in stop_words]

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
    stemmer = PorterStemmer()
    return [stemmer.stem(word) for word in text.split() if word not in stop_words]


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
        Text to be cleaned and tokenised.

    Returns
    -------
    List[str]
        List of cleaned tokens.

    '''
    
    return get_stemmed_tokens(clean_text(text))

if __name__ == "__main__":
    
    test = "</a>This :) is :( a test :-)!"
    test_ = clean_text(test)
    test__ = clean_and_tokenise(test)
    
    # Current working directory
    CWD = os.getcwd()
    
    # Path to data
    data_folder = "Data"
    DATA_PATH = os.path.join(CWD,data_folder)
    
    data_filename="imdb_data.csv"
    df = pd.read_csv(os.path.join(DATA_PATH,data_filename))
    df['Review_text'] = df['Review_text'].apply(clean_text)
    
    