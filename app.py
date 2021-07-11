#Importing Libraries
import pandas as pd
import string
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.stem import WordNetLemmatizer
import joblib
lemma = WordNetLemmatizer()
#Importing Datasets
training_set = pd.read_csv(r'G:\MangeshDataScience\Deployment\Twitter\Twitter-sentiment-prediction\tweet-sentiment-dataset\train.csv')
testing_set = pd.read_csv(r'G:\MangeshDataScience\Deployment\Twitter\Twitter-sentiment-prediction\tweet-sentiment-dataset\test.csv')
training_set.text = training_set.text.astype(str)
training_set.loc[:, 'text'] = training_set.loc[:, 'text'].apply(lambda x: x.lower())
training_set.loc[:, 'text'] = training_set.loc[:, 'text'].apply(lambda x: re.sub(r'@\S+',"",x))
training_set.loc[:, 'text'] = training_set.loc[:, 'text'].apply(lambda x: ' '.join([lemma.lemmatize(word, 'v') for word in nltk.word_tokenize(x) if word not in stopwords.words('english') if word not in string.punctuation]))
testing_set.text = testing_set.text.astype(str)
testing_set.loc[:, 'text'] = testing_set.loc[:, 'text'].apply(lambda x: x.lower())
testing_set.loc[:, 'text'] = testing_set.loc[:, 'text'].apply(lambda x: re.sub(r'@\S+',"",x))
testing_set.loc[:, 'text'] = testing_set.loc[:, 'text'].apply(lambda x: ' '.join([lemma.lemmatize(word, 'v') for word in nltk.word_tokenize(x) if word not in stopwords.words('english') if word not in string.punctuation]))
com_sent = pd.concat([training_set, testing_set], axis = 0).reset_index()
X_train , X_test , y_train , y_test = train_test_split(com_sent['text'].values , com_sent['sentiment'].values , test_size = 0.2, random_state = 101)
train_data = pd.DataFrame({'text':X_train , 'sentiment':y_train})
test_data = pd.DataFrame({'text':X_test , 'sentiment':y_test})
vectorizer = TfidfVectorizer()
train_vector = vectorizer.fit_transform(train_data.text)
test_vector = vectorizer.transform(test_data.text)
from sklearn.metrics import classification_report, accuracy_score
from sklearn import svm
classifier = svm.SVC(kernel='linear')
classifier.fit(train_vector, train_data.sentiment)
joblib.dump(classifier, 'SVMTwitterSent.pkl')
joblib.dump(vectorizer, 'TranformerTwitt.pkl')