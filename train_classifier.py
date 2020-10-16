#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 17:53:49 2020

@author: jkreula
"""
import os
import numpy as np
import pandas as pd
from process_data import clean_text, get_tokens, get_stemmed_tokens, clean_and_tokenise
from get_data import find_num_files
from typing import List, Tuple, Union, Iterator, Generator
from nltk.corpus import stopwords
import pyprind

from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.metrics import roc_auc_score

def stream_files(filepath: str) -> Generator:
    '''
    Create a generator of labels and review text from file.

    Parameters
    ----------
    filepath : str
        DESCRIPTION.

    Yields
    ------
    Generator
        Tuple of label and review text.

    '''
    with open(filepath, r"r", encoding=r"utf-8") as f:
        next(f) # Don't need the column names
        for line in f:
            label, text = line[0], line[3:]
            yield label, text

def get_minibatch(file_stream: Iterator, batch_size: int = 1000) -> Tuple[List[str],List[int]]:
    '''
    Create a minibatch vectors of features and corresponding labels.

    Parameters
    ----------
    file_stream : Iterator
        Generator returned by stream_files function.
    batch_size : int, optional
        Batch size. The default is 1000.

    Returns
    -------
    Tuple[List[str],List[int]]
        Tuple of review text list and corresponding label list.

    '''
    X, y = [], []
    
    try:
        for _ in range(batch_size):
            label, text = next(file_stream)
            y.append(int(label))
            X.append(text)
            
    except StopIteration:
        return None, None
    
    return X, y

def minibatch_partial_fit(stream: Generator, num_batches: int, batch_size: int, vec, clf) -> None:
    
    prog_bar = pyprind.ProgBar(num_batches)
    classes = np.array([0,1])
    
    for _ in range(num_batches):
        
        X_train, y_train = get_minibatch(stream, batch_size)
        
        if X_train is None:
            break
        X_train = vec.transform(X_train)
        clf.partial_fit(X_train, y_train, classes = classes)
        prog_bar.update()
    
    else:
        print(f"\nFitting completed for {num_batches} batches of size {batch_size}.")

if __name__ == "__main__":
    
    online_learning = True
    
    # Current working directory
    CWD = os.getcwd()
    
    # Path to data
    data_folder = "Data"
    DATA_PATH = os.path.join(CWD,data_folder)
    data_filename="imdb_data.csv"

    
    if not online_learning:
            
        # Load data
        df = pd.read_csv(os.path.join(DATA_PATH,data_filename))
        # Clean text
        df['Review_text'] = df['Review_text'].apply(clean_text)
        
        # Train test split
        X = df['Review_text'].values
        y = df['Label'].values
        X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.5,random_state=42)
        
        # Define common stop words
        stop_words = stopwords.words('english')
        
        # Initialise term frequency-inverse document frequency vectorizer
        tfidf = TfidfVectorizer(strip_accents=None,lowercase=False,preprocessor=None)
        
        # Initialise logistic regression classifier
        logreg = LogisticRegression(random_state = 42, solver = 'saga')
        
        
        pipeline_steps = [('tfidf', tfidf),
                          ('clf', logreg)]
        
        # Initialise pipeline
        pipe = Pipeline(pipeline_steps)
        
        # Parameter grid for cross-validation
        param_grid = [{'tfidf__ngram_range': [(1,1),(1,2)],
                       'tfidf__tokenizer': [get_tokens,
                                            get_stemmed_tokens],
                       'clf__penalty': ['l1','l2'],
                       'clf__C': [1e0,1e1,1e2]}]
        
        # Perform grid search cross-validation
        gs = GridSearchCV(pipe, param_grid, scoring='accuracy', cv = 3, verbose = 1, n_jobs = 1)
        gs.fit(X_train,y_train)
        
        # Model evaluation
        print(f"Best parameters: {gs.best_params_}")
        print(f"CV accuracy: {gs.best_score_:.3f}")
        
        logreg_best = gs.best_estimator_
        test_acc = logreg_best.score(X_test,y_test)
        print(f"Test accuracy: {test_acc:.3f}")
        
        y_prob = logreg_best.predict_proba(X_test)
        auc = roc_auc_score(y_test,y_prob[:,1])
        print(f"AUC: {auc:.3f}")
        
    else:
        filepath=os.path.join(DATA_PATH,data_filename)
        stream = stream_files(filepath)
        
        hvec = HashingVectorizer(decode_error = 'ignore',
                                 n_features = 2**21,
                                 preprocessor=None,
                                 tokenizer=clean_and_tokenise)
        
        sgd_clf = SGDClassifier(loss='log',
                                random_state = 42)
        

        minibatch_partial_fit(stream=stream, 
                              num_batches=45, 
                              batch_size=1000, 
                              vec=hvec, 
                              clf=sgd_clf)
        
        X_test, y_test = get_minibatch(stream, batch_size = 5000)
        X_test = hvec.transform(X_test)
        print(f"Accuracy {sgd_clf.score(X_test,y_test):.3f}")