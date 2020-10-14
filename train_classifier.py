#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 17:53:49 2020

@author: jkreula
"""
import os
import numpy as np
import pandas as pd
from process_data import clean_text, get_tokens, get_stemmed_tokens
from typing import List, Tuple, Union
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.metrics import roc_auc_score

# Current working directory
CWD = os.getcwd()

# Path to data
data_folder = "Data"
DATA_PATH = os.path.join(CWD,data_folder)
data_filename="imdb_data.csv"
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

print(f"Best parameters: {gs.best_params_}")
print(f"CV accuracy: {gs.best_score_:.3f}")

logreg_best = gs.best_estimator_
test_acc = logreg_best.score(X_test,y_test)
print(f"Test accuracy: {test_acc:.3f}")

y_prob = logreg_best.predict_proba(X_test)
auc = roc_auc_score(y_test,y_prob[:,1])
print(f"AUC: {auc:.3f}")
