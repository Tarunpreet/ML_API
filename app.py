from flask import Flask
from flask_cors import CORS, cross_origin
from flask_restful import reqparse, abort, Api, Resource
from sklearn.externals import joblib
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_extraction.text import TfidfTransformer 
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import preprocessing
import pandas as pd


app = Flask(__name__)
api = Api(app)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'


tfidf = joblib.load('tfidf_joblib')

# parser = reqparse.RequestParser()
# parser.add_argument('query')

class PredictSentiment(Resource):
    def get(self,query):
        if query=='cong':
            test=pd.read_csv('https://docs.google.com/spreadsheets/d/e/2PACX-1vR5cg9wx36ZoCO3BK4ePB_OCA3kFphDxmRaxoGHqggsdsn3p2ziOO0Ss3CvwuSb3IAbL3Btj5Vc_SI_/pub?output=csv')
            #test=pd.read_csv('test1.csv', encoding='Windows-1252')
            X_test=test['SentimentText'].values
            X_test = tfidf.transform(X_test)
            X_test = preprocessing.normalize(X_test)
            X_test = X_test.todense()
        elif query=='bjp':
            test=pd.read_csv('https://docs.google.com/spreadsheets/d/e/2PACX-1vSKmNP2Au_O1LTxo1rSd4NaWTSFCNDHx6a25hQzohcbZ-BH9yBjbJW92wMUm4S_mrpLV0gw2nuHmwkS/pub?output=csv')
            #test=pd.read_csv('test1.csv', encoding='Windows-1252')
            X_test=test['SentimentText'].values
            X_test = tfidf.transform(X_test)
            X_test = preprocessing.normalize(X_test)
            X_test = X_test.todense()
        jb=joblib.load('model_joblib')
        pred=jb.predict(X_test)
        positive=0
        negative=0
        for i in pred:
            if i==0:    
                negative=negative+1
            else:
                positive=positive+1
        if positive > negative:
            pred='positive'
        else:
            pred='negative'
        positive_perc=(float(positive)/(positive+negative))*100
        negative_perc=(float(negative)/(positive+negative))*100
        output={'prediction':pred,'positive_no':positive,'negative_no':negative,'positive_per':positive_perc,'negative_per':negative_perc,'party':query}
        return output

api.add_resource(PredictSentiment, '/<string:query>')

if __name__ == '__main__':
    app.run(debug=True)

