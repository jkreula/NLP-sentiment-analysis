#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 18 12:01:00 2020

@author: jkreula
"""

import sqlite3
import os

conn = sqlite3.connect("reviews.sqlite")

c = conn.cursor()

c.execute("DROP TABLE IF EXISTS review_db")
c.execute("CREATE TABLE review_db (Label INTEGER, Review_text TEXT, date TEXT)")

example1 = "This movie is great! Go watch it!"

c.execute("INSERT INTO review_db (Label, Review_text, date) VALUES (?, ?, DATETIME('now'))", (1, example1))

example2 = "I did not like this movie. It's awful."

c.execute("INSERT INTO review_db (Label, Review_text, date) VALUES (?, ?, DATETIME('now'))", (0, example2))

conn.commit()
conn.close()

conn = sqlite3.connect("reviews.sqlite")
c = conn.cursor()
command = "SELECT * FROM review_db"\
          " WHERE date"\
          " BETWEEN '2017-01-01 00:00:00' AND DATETIME('now')"
c.execute(command)
results = c.fetchall()
conn.close()
print(results)