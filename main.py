#import data
import pandas as pd
import numpy as np
import nltk
import re
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from imblearn.combine import SMOTETomek
from sklearn.metrics import accuracy_score, confusion_matrix,classification_report
data=pd.read_excel('1000 leads.xlsx')


data['Status '].replace(to_replace="NOt Converted",value="Not Converted",inplace=True)
data['Status '].replace(to_replace='Converted ',value="Converted",inplace=True)
data['Status '].replace(to_replace='Conveted',value="Converted",inplace=True)


data.drop(columns=['Lead Name'],inplace=True)
data.dropna(inplace=True)
data.reset_index(inplace=True)
df=data
from nltk.corpus import stopwords
#from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
nltk.download(['wordnet','stopwords'] )
#ps = PorterStemmer()
wordnet=WordNetLemmatizer()
corpus = []

for i in range(0,len(data)):
    review=re.sub('[^a-zA-Z]',' ',data['Status information'][i])
    review=review.lower()
    review=review.split()

    review = [wordnet.lemmatize(word) for word in review if not word in stopwords.words('english')] 
    # removing stop words and also lematizing
    review=" ".join(review)
    corpus.append(review)


for i in range(0 ,len(corpus)):
    if "prema" in corpus[i]:
        corpus[i]=corpus[i].replace("prema",'',True)
    elif "mohan" in corpus[i]:
        corpus[i]=corpus[i].replace("mohan",'',True)
    elif "gowtham" in corpus[i]:
        corpus[i]=corpus[i].replace("gowtham",'',True)
    elif "surendra" in corpus[i]:
        corpus[i]=corpus[i].replace("surendra",'',True)
    elif "soma" in corpus[i]:
        corpus[i]=corpus[i].replace("soma",'',True)
    else:
        corpus[i]=corpus[i]


#cv = CountVectorizer(max_features=2500)
#X = cv.fit_transform(corpus).toarray()

tv_t1 = TfidfVectorizer(max_features=3000,ngram_range=(3,6))      
X = tv_t1.fit_transform(corpus).toarray()      

y=pd.get_dummies(data['Status ']) #onehot coding for label column into y variable
# Not converted : 1
#converted :0
y=y.iloc[:,1].values

smk = SMOTETomek(random_state=1234)
x_new, y_new = smk.fit_resample(X,y)
x_new.shape, y_new.shape


X_train, X_test, y_train, y_test = train_test_split(x_new, y_new, test_size = 0.20, random_state = 0)

model_t1 = SVC(kernel = 'rbf', random_state = 0).fit(X_train, y_train)
y_pred=model_t1.predict(X_test)
print(accuracy_score(y_pred, y_test))
print(confusion_matrix(y_pred, y_test))
print(classification_report(y_pred,y_test))


import pickle
pickle.dump(model_t1, open("model_t1.pkl", "wb") )


import pickle
pickle.dump(tv_t1, open("tv_t1.pkl", "wb"))

corpus1 = []

for i in range(0,len(df)):
    review1=re.sub('[^a-zA-Z]',' ',df['Status information'][i])
    review1=review1.lower()
    review1=review1.split()

    review1 = [wordnet.lemmatize(word) for word in review1 if not word in stopwords.words('english')] 
    # removing stop words and also lematizing
    review1=" ".join(review1)
    corpus1.append(review1)

corpus2 = []

for i in range(0,len(df)):
    review2=re.sub('[^a-zA-Z]',' ',df['Location'][i])
    review2=review2.lower()
    review2=review2.split()

    review2 = [wordnet.lemmatize(word) for word in review2 if not word in stopwords.words('english')] 
    # removing stop words and also lematizing
    review2=" ".join(review2)
    corpus2.append(review2)

BE=[]
for i in range(0 ,len(corpus1)):
    if "prema" in corpus1[i]:
        BE.append('prema')
        corpus1[i]=corpus1[i].replace("prema",'',True)
    elif "mohan" in corpus1[i]:
        BE.append('mohan')
        corpus1[i]=corpus1[i].replace("mohan",'',True)
    elif "gowtham" in corpus1[i]:
        BE.append('gowtham')
        corpus1[i]=corpus1[i].replace("gowtham",'',True)
    elif "surendra" in corpus1[i]:
        BE.append('surendra')
        corpus1[i]=corpus1[i].replace("surendra",'',True)
    elif "soma" in corpus1[i]:
        BE.append('soma')
        corpus1[i]=corpus1[i].replace("soma",'',True)
    else:
        BE.append('others')
        corpus1[i]=corpus1[i]

len(BE)
df['BE']=BE

corpus3 = []

for i in range(0,len(df)):
    review3=re.sub('[^a-zA-Z]',' ',df['BE'][i])
    review3=review3.lower()
    review3=review3.split()

    review3 = [wordnet.lemmatize(word) for word in review3 if not word in stopwords.words('english')] 
    # removing stop words and also lematizing
    review3=" ".join(review3)
    corpus3.append(review3)

#cv = CountVectorizer(max_features=2500)
#X = cv.fit_transform(corpus).toarray()

tf1 = TfidfVectorizer(max_features=3000,ngram_range=(3,6))      #defining tfidf vec for transformation
x1 = tf1.fit_transform(corpus1).toarray()#transforming our preprocesssed data to vectors adn array
tf2 = TfidfVectorizer(max_features=3000,ngram_range=(1,2))      #defining tfidf vec for transformation
x2 = tf2.fit_transform(corpus2).toarray()
tf3 = TfidfVectorizer(max_features=3000,ngram_range=(1,2))
x3 = tf3.fit_transform(corpus3).toarray()


y=pd.get_dummies(df['Status ']) #onehot coding for label column into y variable
# Not converted : 1
#converted :0
y=y.iloc[:,1].values 

x=np.concatenate((x1,x2,x3),axis=1)

smk = SMOTETomek(random_state=1234)
x, y = smk.fit_resample(x,y)
x.shape, y.shape

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.20, random_state = 0)

model_t2 = SVC(kernel = 'rbf', random_state = 0).fit(X_train, y_train)
y_pred=model_t2 .predict(X_test)
print(accuracy_score(y_pred, y_test))
print(confusion_matrix(y_pred, y_test))
print(classification_report(y_pred,y_test))

import pickle
pickle.dump(model_t2, open("model_t2.pkl", "wb") )


import pickle
pickle.dump(tf1, open("tf1.pkl", "wb"))
pickle.dump(tf2, open("tf2.pkl", "wb"))
pickle.dump(tf3, open("tf3.pkl", "wb"))








