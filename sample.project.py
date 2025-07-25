import chardet
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
with open('spam.csv', 'rb') as f:
    result = chardet.detect(f.read())
df = pd.read_csv('spam.csv', encoding=result['encoding'], engine='python')
df.head()
df.drop(columns=['Unnamed: 2','Unnamed: 3','Unnamed: 4'],inplace=True)
df.rename(columns={'v1':'Category','v2':'Message'},inplace=True)
df.shape
df.drop_duplicates(inplace=True)
df.shape
df.isnull().sum()
df['Category'] = df['Category'].replace(['ham','spam'],['Not Spam','Spam'])
df.head()
X = df['Message']
Y = df['Category']
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2)
cv = CountVectorizer(stop_words='english')
features = cv.fit_transform(X_train)
# Creatig Model
model = MultinomialNB()
model.fit(features,Y_train)
# Testing Model
features_test = cv.transform(X_test)
model.score(features_test,Y_test)
# Predict Data
message = cv.transform(['msaadraza06@gmail.com']).toarray()
result = model.predict(message)
print(result)