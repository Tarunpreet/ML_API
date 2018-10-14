from flask import Flask
from flask_restful import reqparse, abort, Api, Resource
from sklearn.externals import joblib
# from sklearn.cross_validation import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
#from sklearn.cross_validation import cross_val_score
from collections import Counter
from sklearn.metrics import accuracy_score
#from sklearn import cross_validation
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_extraction.text import TfidfTransformer 
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score,f1_score,recall_score
from sklearn import preprocessing
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd


app = Flask(__name__)
api = Api(app)


tfidf = joblib.load('tfidf_joblib')
test=pd.read_csv('https://docs.google.com/spreadsheets/d/e/2PACX-1vR5cg9wx36ZoCO3BK4ePB_OCA3kFphDxmRaxoGHqggsdsn3p2ziOO0Ss3CvwuSb3IAbL3Btj5Vc_SI_/pub?output=csv')
#test=pd.read_csv('test1.csv', encoding='Windows-1252')
X_test=test['SentimentText'].values
X_test = tfidf.transform(X_test)
X_test = preprocessing.normalize(X_test)
X_test = X_test.todense()
jb=joblib.load('model_joblib')

parser = reqparse.RequestParser()

def posneg():
    if Positive > Negative:
        pred='positive'
    else:
        pred='negative'
class PredictSentiment(Resource):
	def get(self):
		pred=jb.predict(X_test)
		Negative = (pred == 0).sum()
		Positive = (pred == 1).sum()
        # create JSON object
        posneg()
        output = {'prediction': pred, 'positive_rate': Positive,'Negative_rate': Negative }
        return output

api.add_resource(PredictSentiment, '/')

if __name__ == '__main__':
    app.run(debug=True)

