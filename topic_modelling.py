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
from time import time

def print_topics(model, feature_names, num_words) -> None:
    for topic_idx, topic in enumerate(model.components_, 1):
        print(f"Topic #{topic_idx}:")
        print(" ".join([feature_names[i] for i in topic.argsort()[:-num_words-1:-1]]))
        print()


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
    
    n_components = 10
    
    # LDA
    print("Start LDA")
    time_start_lda = time()
    lda = LatentDirichletAllocation(n_components = n_components,
                                    random_state = 123,
                                    learning_method = 'online')
    X_topics_LDA = lda.fit_transform(X)
    print(f"LDA finished in {time()-time_start_lda:.2f}s.")
    
    num_words = 5
    
    feature_names = count_vec.get_feature_names()

    print("Topics for LDA")
    print_topics(lda, feature_names, num_words)

    # NMF
    print("Start NMF")
    time_start_nmf = time()
    nmf = NMF(n_components=n_components, 
              random_state=123,
              alpha=0.1, 
              l1_ratio=0.5,
              max_iter = 500)
    X_topics_NMF = nmf.fit_transform(X)
    print(f"NMF finished in {time()-time_start_nmf:.2f}s.")
    
    print("Topics for NMF")
    print_topics(nmf, feature_names, num_words)