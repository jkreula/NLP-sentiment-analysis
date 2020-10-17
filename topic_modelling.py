#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 17 02:47:22 2020

@author: jkreula
"""
import os
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF

if __name__ == "__main__":
    # Current directory
    CWD = os.path.dirname(__file__)
    
    # Path to data
    data_folder = "Data"
    DATA_PATH = os.path.join(CWD,data_folder)
    data_filename="imdb_data.csv"
    
    filepath=os.path.join(DATA_PATH,data_filename)
    
    df = pd.read_csv(filepath, encoding = r"utf-8")
    
    count_vec = CountVectorizer(stop_words = 'english',
                                max_df = 0.1,
                                max_features = 5000)
    
    X = count_vec.fit_transform(df['Review_text'].values)
    
    lda = LatentDirichletAllocation(n_components = 10,
                                    random_state = 123,
                                    learning_method = 'batch')
    
    X_topics = lda.fit_transform(X)