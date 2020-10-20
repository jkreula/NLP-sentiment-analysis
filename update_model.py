#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 20:34:52 2020

@author: jkreula
"""
import pickle
import sqlite3
import numpy as np
import os
from datetime import date

from vectorizer import hvec

def update_model(*, database_path: str, model, batch_size: int = int(1e4)):
    
    conn = sqlite3.connect(database_path)
    c = conn.cursor()
    c.execute('SELECT * from review_db')
    
    results = c.fetchmany(batch_size)
    
    while len(results) > 0:
        data = np.array(results)
        # Extract features and labels
        X = data[:, 1]
        y = data[:, 0].astype(int)
        
        # Available class labels
        classes = np.array([0, 1])
        
        # Apply hashing vectorizer
        X_train = hvec.transform(X)
        # Update model
        model.partial_fit(X_train, y, classes = classes)
        
        # Next batch
        results = c.fetchmany(batch_size)
        
    conn.close()
    return model

current_dir = os.path.dirname(__file__)

clf = pickle.load(open(os.path.join(current_dir,'pickled_objects','classifier.pkl'),'rb'))

update_pickle_file = False
add_date = True

if update_pickle_file:
    filename = 'classifier' + date.today().strftime('%Y%m%d') + '.pkl' if add_date else 'classifier.pkl'
    pickle.dump(clf,open(os.path.join(current_dir,'pickled_objects',),'wb'), protocol = 5)
    
