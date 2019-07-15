from flask import Flask,render_template,url_for,request,jsonify
import pandas as pd 
import numpy as np
import json
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.externals import joblib

request = 1
msg = "hai hello hw are yoi"

def predict():
	#msg = request.args.get('msg')
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
	#naive bayes classsifier
	from sklearn.naive_bayes import MultinomialNB

	clf = MultinomialNB()
	clf.fit(X_train,y_train)
	clf.score(X_test,y_test)
	#not working -flag
	#Alternative Usage of Saved Model
	# joblib.dump(clf, 'NB_spam_model.pkl')
	# NB_spam_model = open('NB_spam_model.pkl','rb')
	# clf = joblib.load(NB_spam_model)
   
	if request  == 1:
		file1 = open("data.txt","w+") 
		file1.write("hai hello \n hai hello \n hello")
		data = file1
		vect = cv.transform(data).toarray()
		my_prediction = clf.predict(vect)
		file1.close()
		if np.any(my_prediction) == 1:
			res = "hai"
		else:
			res = "hello"	
	return res

predict()