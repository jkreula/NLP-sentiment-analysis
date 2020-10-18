#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 18 16:04:26 2020

@author: jkreula
"""

from flask import Flask, render_template, request
from wtforms import Form, TextAreaField, validators
import pickle
import sqlite3
import os
import numpy as np

from vectorizer import hvec
from typing import Tuple

app = Flask(__name__)

current_dir = os.path.dirname(__file__)
clf = pickle.load(open(os.path.join(current_dir,"pickled_objects","online_classifier.pkl"),"rb")) 

database_path = os.path.join(current_dir, "reviews.sqlite")

def classify(text: str) -> Tuple[int, float]:
    label = {0: 'negative', 1: 'positive'}
    X = hvec.transform([text])
    y = clf.predict(X)[0]
    prob = np.max(clf.predict_proba(X))
    return label[y], prob

def train_model_online(text: str, y: int) -> None:
    X = hvec.transform([text])
    clf.partial_fit(X,[y])
    
    
def database_entry(path: str, text: str, y: int) -> None:
    conn = sqlite3.connect(path)
    c = conn.cursor()
    c.execute("INSERT INTO review_db (Label, Review_text, date) VALUES (?, ?, DATETIME('now'))", (y, text))
    conn.commit()
    conn.close()
    

# Flask
class ReviewForm(Form):
    moviereview = TextAreaField('',[validators.DataRequired(),
                               validators.length(min=15)])
    
@app.route('/')
def index():
    form = ReviewForm(request.form)
    return render_template('reviewform.html', form = form)

@app.route('/results', methods=['POST'])
def results():
    form = ReviewForm(request.form)
    if request.method == 'POST' and form.validate():
        review = request.form['moviereview']
        y, prob = classify(review)
        return render_template('results.html',
                               content=review,
                               prediction=y,
                               probability=round(prob*100,2))
    return render_template('reviewform.html',form=form)

@app.route('/thanks', methods=['POST'])
def feedback():
    feedback = request.form['feedback_button']
    review = request.form['review']
    prediction = request.form['prediction']
    
    inv_label = {'negative': 0, 'positive': 1}
    y = inv_label[prediction]
    if feedback == 'Incorrect':
        y = int(not(y))
    train_model_online(review,y)
    database_entry(database_path, review, y)
    return render_template('thanks.html')

if __name__ == "__main__":
    app.run(debug=True)
    #review_txt = "Awesome movie, really witty and hilarious!"
    #database_entry(database_path, review_txt, 1)
   