#!/usr/bin/env python
# coding: utf-8

# ## Importing the relevant libraries

# In[1]:


import joblib, requests, string
from joblib import dump, load
import pandas as pd
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk import word_tokenize
import math, nltk, json
from collections import Counter
from flask import Flask, jsonify, render_template, request


# ## Initializing the Flask app

# In[2]:


app = Flask(__name__)


# ## Functions for text preprocessing

# In[3]:


stemmer = PorterStemmer()
def stem_words(text):
    return ' '.join([stemmer.stem(word) for word in text.split()])

lemmatizer = WordNetLemmatizer()
def lemmatize_words(text):
    return ' '.join([lemmatizer.lemmatize(word) for word in text.split()])

def text_process(text):
    nopunc = [char for char in text if char not in string.punctuation]
    nopunc = ''.join(nopunc)
    return ' '.join([word for word in nopunc.split() if word.lower() not in stopwords.words('english') and not word.isdigit()])


# ## Function for creating vectors for each of the two texts:

# In[4]:


def build_vector(iterable1, iterable2):
    counter1 = Counter(iterable1)
    counter2 = Counter(iterable2)
    all_items = set(counter1.keys()).union(set(counter2.keys()))
    vector1 = [counter1[k] for k in all_items]
    vector2 = [counter2[k] for k in all_items]
    return vector1, vector2


# ## Function for calculating cosine similarity between the two vectors based on the angle between them in a multi-dimensional space:

# Cosine Similarity measures the cosine of the angle between two embeddings. When the embeddings are pointing in the same direction the angle between them is zero so their cosine similarity is 1 when the embeddings are orthogonal the angle between them is 90 degrees and the cosine similarity is 0 finally when the angle between them is 180 degrees the the cosine similarity is -1.

# Mathematically, we can calculate the cosine similarity by taking the dot product between the embeddings and dividing it by the multiplication of the embeddings norms.
# 

# ![cosine similarity](cosine_similarity.png)

# In[5]:


def cosim(v1, v2):
    dot_product = sum(n1 * n2 for n1, n2 in zip(v1, v2) )
    magnitude1 = math.sqrt(sum(n ** 2 for n in v1))
    magnitude2 = math.sqrt(sum(n ** 2 for n in v2))
    return dot_product / (magnitude1 * magnitude2)


# ## App Routing Code for Model Deployment:

# In[ ]:


@app.route('/')
def home():
    return render_template('text_similarity.html')

@app.route('/predict',methods=['POST'])
def predict():
    if request.method == 'POST':
        text1 = request.form.get("text1")
        text2 = request.form.get("text2")
#         text1 = text_process(text1)
#         text2 = text_process(text2)
#         text1 = stem_words(text1)
#         text2 = stem_words(text2)
#         text1 = lemmatize_words(text1)
#         text2 = lemmatize_words(text2)
        t1 = text1.split()
        t2 = text2.split()
        v1,v2 = build_vector(t1,t2)
        return jsonify({'similarity_score': cosim(v1,v2)})

if __name__ == '__main__':
    cosine_similarity_model = joblib.load('model.pkl')
    app.run(port=8080)

