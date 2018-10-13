
# coding: utf-8

# In[2]:


#get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings("ignore")

import sqlite3
import pandas as pd
import numpy as np
import nltk
import string
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.externals import joblib
from sklearn.metrics import roc_curve, auc
from nltk.stem.porter import PorterStemmer

import re
# Tutorial about Python regular expressions: https://pymotw.com/2/re/
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer

from gensim.models import Word2Vec
from gensim.models import KeyedVectors
import pickle


# In[3]:


#Load train.csv into a pandas dataFrame.
train = pd.read_csv('train.csv', encoding='Windows-1252')


# In[4]:


print (train.shape)
print (train.columns)


# In[5]:


def partition(x):
    if x == 0:
        return 'negative'
    return 'positive'

#changing reviews with sentiment 1 to be positive and vice-versa

actualScore = train['Sentiment']
positiveNegative = actualScore.map(partition) 
train['Sentiment'] = positiveNegative


# In[6]:

#looking at the number of attributes and size of the data

print(train.shape) 

train.head()


# ##   Text Preprocessing: Stemming, stop-word removal and Lemmatization.
# 
# Now that we have finished deduplication our data requires some preprocessing before we go on further with analysis and making the prediction model.
# 
# Hence in the Preprocessing phase we do the following in the order below:-
# 
# 1. Begin by removing the html tags
# 2. Remove any punctuations or limited set of special characters like , or . or # etc.
# 3. Check if the word is made up of english letters and is not alpha-numeric
# 4. Check to see if the length of the word is greater than 2 (as it was researched that there is no adjective in 2-letters)
# 5. Convert the word to lowercase
# 6. Remove Stopwords
# 7. Finally Snowball Stemming the word (it was obsereved to be better than Porter Stemming)<br>
# 
# After which we collect the words used to describe positive and negative reviews

# In[7]:


# find sentences containing HTML tags
import re
i=0;
for sent in train['SentimentText'].values:
    if (len(re.findall('<.*?>', sent))):
        print(i)
        print(sent)
        break;
    i += 1;


# In[8]:


import nltk
nltk.download('stopwords')
  
stop = set(stopwords.words('english')) #set of stopwords
sno = nltk.stem.SnowballStemmer('english') #initialising the snowball stemmer

def cleanhtml(sentence): #function to clean the word of any html-tags
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, ' ', sentence)
    return cleantext
def cleanpunc(sentence): #function to clean the word of any punctuation or special characters
    cleaned = re.sub(r'[?|!|\'|"|#]',r'',sentence)
    cleaned = re.sub(r'[.|,|)|(|\|/]',r' ',cleaned)
    return  cleaned
print(stop)
print('************************************')
print(sno.stem('tasty'))


# In[9]:


#Code for implementing step-by-step the checks mentioned in the pre-processing phase
# this code takes a while to run as it needs to run on 100k sentences.
i=0
str1=' '
final_string=[]
all_positive_words=[] # store words from +ve reviews here
all_negative_words=[] # store words from -ve reviews here.
s=''
for sent in train['SentimentText'].values:
    filtered_sentence=[]
    print(sent);
    sent=cleanhtml(sent) # remove HTMl tags
    for w in sent.split():
        for cleaned_words in cleanpunc(w).split():
            if((cleaned_words.isalpha()) & (len(cleaned_words)>2)):    
                if(cleaned_words.lower() not in stop):
                    s=(sno.stem(cleaned_words.lower())).encode('utf8')
                    filtered_sentence.append(s)
                    if (train['Sentiment'].values)[i] == 'positive': 
                        all_positive_words.append(s) #list of all words used to describe positive reviews
                    if(train['Sentiment'].values)[i] == 'negative':
                        all_negative_words.append(s) #list of all words used to describe negative reviews reviews
                else:
                    continue
            else:
                continue 
    print(filtered_sentence)
    str1 = b" ".join(filtered_sentence) #final string of cleaned words
    print("***********************************************************************")
    
    final_string.append(str1)
    i+=1


# In[10]:


train['CleanedText']=final_string #adding a column of CleanedText which displays the data after pre-processing of the review 
train['CleanedText']=train['CleanedText'].str.decode("utf-8")


# In[11]:


# now make classification label for Score label
def make_0_1(x):
    if x=='positive':
        return 1 
    else:
        return 0


# In[12]:


train['Sentiment']= train['Sentiment'].map(make_0_1)


# In[13]:


train.head(3) 
#below the processed review can be seen in the CleanedText Column 


# In[14]:


import random
train=train.iloc[random.sample(range(len(train)), 30000)]
print(train.shape)


#  Naive Bayes using TF-IDF

# In[31]:


from sklearn.cross_validation import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import cross_val_score
from collections import Counter
from sklearn.metrics import accuracy_score
from sklearn import cross_validation
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


# In[32]:

# test=pd.read_csv('https://docs.google.com/spreadsheets/d/e/2PACX-1vR5cg9wx36ZoCO3BK4ePB_OCA3kFphDxmRaxoGHqggsdsn3p2ziOO0Ss3CvwuSb3IAbL3Btj5Vc_SI_/pub?output=csv')
# X_train=train['CleanedText'].values;
# y_train=train['Sentiment'].values;
# X_test=test['SentimentText'].values;
# y_test=test['Sentiment'].values
X_train,X_test,y_train,y_test = train_test_split(train['CleanedText'].values,train['Sentiment'].values,test_size=0.2,random_state=42)


# In[33]:


tfidf = TfidfVectorizer(ngram_range=(1,2))

# tranfomration and normalization
X_train = tfidf.fit_transform(X_train) 
X_train = preprocessing.normalize(X_train)
joblib.dump(tfidf,'tfidf_joblib')
# tranfomration and normalization
X_test = tfidf.transform(X_test)
X_test = preprocessing.normalize(X_test)


# In[34]:


print(X_train.shape)
print(X_test.shape)
print(type(X_train))
print(type(y_train))


# In[35]:


X_train = X_train.todense()
X_test = X_test.todense()


# In[36]:


# creating  list of alpha 

neighbors = [0.0001,0.001,0.01,0.1, 1,10,100,1000]

# empty list that will hold cv scores
cv_scores = []

for hyperparameter in neighbors:
    clf = MultinomialNB()
    model = clf.fit(X_train,y_train)
    joblib.dump(model,'model_joblib')
    pred = model.predict(X_test)
    acc = accuracy_score(y_test,pred,normalize=True)
    cv_scores.append(acc)

for i in pred:
    print(i)
    print(" ")
   
    

#changing to misclassification error
MSE = [1 - x for x in cv_scores]

#determining best alpha
optimal_k = neighbors[MSE.index(min(MSE))]

print('\nThe optimal value of alpha is %f.' % optimal_k)

print("Misclassification error for each hyperparameter value is  : ", np.round(MSE,3))


# In[37]:


clf = GaussianNB()
model = clf.fit(X_train,y_train)


# In[38]:


precision = precision_score(y_test, pred)
print("Precision on test set: %0.3f"%(precision)) 
recall = recall_score(y_test, pred)
print("Recall on test set: %0.3f"%(recall)) 
f1 = f1_score(y_test, pred)
print("F1-Score on test set: %0.3f"%(f1))


# In[39]:


y = confusion_matrix(y_test,pred)
y


# In[40]:


True_Negative = y[0][0]
False_Negative = y[0][1]
False_Positive = y[1][0]
True_Positive = y[1][1]

Total_Negative = True_Negative + False_Positive
Total_Positive = False_Negative + True_Positive

TPR = float(True_Positive/Total_Positive)
print("The True Positive Rate is ", TPR)

TNR = float(True_Negative/Total_Negative)
print("The True Negative Rate is ", TNR)

FNR = float(False_Negative/Total_Positive)
print("The False Negative Rate is ", FNR)

FPR = float(False_Positive/Total_Negative)
print("The False Positive Rate is ", FPR)


# In[41]:


def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()


# In[42]:


plot_confusion_matrix(cm           = y, 
                      normalize    = False,
                      target_names = ['high', 'low'],
                      title        = "Confusion Matrix")


# In[43]:


print (pred)
Negative = (pred == 0).sum()
Positive = (pred == 1).sum()
print("Number of Positive Reviews:")
print(Positive)
print("Number of Negative Reviews:")
print(Negative)


# Conclusions

#In[44]:


results=pd.DataFrame(columns=['vectorization_method', 'Accuracy', 'Alpha', 'Precision', 'Recall', 'F1-Score'])


#In[45]:


results=results.append(
    [
        {
            'vectorization_method' : 'TF-IDF',
            'Accuracy' : acc,
            'Alpha': optimal_k,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1,
            'TPR': TPR,
            'TNR': TNR,
            'FNR': FNR,
            'FPR': FPR     
        },
    ]
)


# In[46]:


results

