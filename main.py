import flask
from flask import Flask, request, render_template
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import joblib
import numpy as np

app = Flask(__name__)
classifier = joblib.load('SVMTwitterSent.pkl')
vectorizer = joblib.load('TranformerTwitt.pkl')

@app.route('/')
def welcome():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    
    sentiment = request.form.get('string')
    tfidf = vectorizer.transform([sentiment])
    pred = classifier.predict(tfidf)
    return render_template('index.html', prediction_text= pred[0])

if __name__=='__main__':
    app.run(debug=True)