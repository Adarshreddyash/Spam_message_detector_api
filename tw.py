from flask import Flask,render_template,url_for,request , jsonify
import pandas as pd 
import numpy as np
import json
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.externals import joblib

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict/<msg>')
def predict(msg):
	df= pd.read_csv("spam.csv", encoding="latin-1")
	df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, inplace=True)
	# Features and Labels
	df['label'] = df['class'].map({'ham': 0, 'spam': 1})
	X = df['message']
	y = df['label']
	
	# Extract Feature With CountVectorizer
	cv = CountVectorizer()
	X = cv.fit_transform(X) # Fit the Data
	from sklearn.model_selection import train_test_split
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
	#Naive Bayes Classifier
	from sklearn.naive_bayes import MultinomialNB

	clf = MultinomialNB()
	clf.fit(X_train,y_train)
	clf.score(X_test,y_test)
	file1 = open('ash.txt','w+')
	file1.write(str(msg))
	data = file1
	vect = cv.transform(data).toarray()
	my_prediction = clf.predict(vect)
	#print(my_prediction)
	if np.any(my_prediction) == 1:
		return jsonify("y")
	elif np.any(my_prediction) == 0:
		return jsonify("naa")

if __name__ == '__main__':
	app.run(debug=True)