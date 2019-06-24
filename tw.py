import pandas as pd 
import numpy as np
import pickle
import json
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.externals import joblib



def predict(request):
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
	#Alternative Usage of Saved Model
	# joblib.dump(clf, 'NB_spam_model.pkl')
	# NB_spam_model = open('NB_spam_model.pkl','rb')
	# clf = joblib.load(NB_spam_model)

	if request == 1:
		file1 = open("ash.txt","r")
		data = file1
		vect = cv.transform(data).toarray()
		my_prediction = clf.predict(vect)
		print(my_prediction)
		if np.any(my_prediction) == 1:
			print("shut that's a spam")
		else:
			print("not a spam !")

		
predict(1)
		