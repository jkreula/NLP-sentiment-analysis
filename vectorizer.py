#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 18 11:27:47 2020

@author: jkreula
"""

from sklearn.feature_extraction.text import HashingVectorizer
import re
import os
import pickle

current_dir = os.path.dirname(__file__)

stop_words = pickle.load(open(os.path.join(current_dir,"pickled_objects","stopwords.pkl"),"rb"))

def tokeniser(text):
    
    text = re.sub('<[^>]*>','',text)
    smileys = re.findall(r'(?::|;|=)(?:-)?(?:\)|\(|D|P)',text)
    text = re.sub(r'[\W]+',' ',text.lower()) + ' '.join(smileys).replace('-','')
    tokens = [word for word in text.split() if word not in stop_words]
    return tokens

hvec = HashingVectorizer(decode_error = 'ignore',
                                 n_features = 2**21,
                                 preprocessor=None,
                                 tokenizer=tokeniser)