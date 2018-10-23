from sklearn.externals import joblib
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_extraction.text import TfidfTransformer 
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import preprocessing
import pandas as pd

print('1. Python Reviews')
print('2. Java Reviews')
print('3. Comparison')

options=input('Enter the option: ')

tfidf = joblib.load('tfidf_joblib')

if options=='1':
    test=pd.read_csv('https://docs.google.com/spreadsheets/d/e/2PACX-1vR5cg9wx36ZoCO3BK4ePB_OCA3kFphDxmRaxoGHqggsdsn3p2ziOO0Ss3CvwuSb3IAbL3Btj5Vc_SI_/pub?output=csv')
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
        pred2='positive'
    else:
        pred2='negative'
    positive_perc=(float(positive)/(positive+negative))*100
    negative_perc=(float(negative)/(positive+negative))*100
    print("The Review is {}".format(pred2))
    print("The Positive Percentage is {}".format(positive_perc))
    print("The Positive Percentage is {}".format(negative_perc))

elif options=='2':
    test1=pd.read_csv('https://docs.google.com/spreadsheets/d/e/2PACX-1vSKmNP2Au_O1LTxo1rSd4NaWTSFCNDHx6a25hQzohcbZ-BH9yBjbJW92wMUm4S_mrpLV0gw2nuHmwkS/pub?output=csv')
    X_test1=test1['SentimentText'].values
    X_test1 = tfidf.transform(X_test1)
    X_test1 = preprocessing.normalize(X_test1)
    X_test1 = X_test1.todense()
    jb=joblib.load('model_joblib')
    pred1=jb.predict(X_test1)
    positive1=0
    negative1=0
    for i in pred1:
        if i==0:
            negative1=negative1+1
        else:
            positive1=positive1+1
    if positive1 > negative1:
        pred3='positive'
    else:
        pred3='negative'
    positive_perc1=(float(positive1)/(positive1+negative1))*100
    negative_perc1=(float(negative1)/(positive1+negative1))*100
    print("The Review is {}".format(pred3))
    print("The Positive Percentage is {}".format(positive_perc1))
    print("The Negative Percentage is {}".format(negative_perc1))
elif options=='3':
	print('Comparison')
