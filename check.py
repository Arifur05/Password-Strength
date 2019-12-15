import random
from warnings import simplefilter

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

simplefilter(action='ignore', category=FutureWarning)

password_data = pd.read_csv("password.csv")
password_data.head()
print(password_data.shape)

password = np.array(password_data)

random.shuffle(password)
ylabels = [s[1] for s in password]
allpasswords = [s[0] for s in password]

ylabels
len(ylabels)
len(allpasswords)


def makeTokens(f):
    tokens = []
    for i in f:
        tokens.append(i)
    return tokens


vectorizer = TfidfVectorizer(tokenizer=makeTokens)

X = vectorizer.fit_transform(allpasswords)

X_train, X_test, Y_train, Y_test = train_test_split(X, ylabels, test_size=0.2, random_state=42)
print(X_train.shape)
print(X_test.shape)

logit = LogisticRegression(penalty='l1', multi_class='ovr')
logit.fit(X_train, Y_train)
print("Accuracy: ", logit.score(X_test, Y_test))

X_predict = ['shudhupapssword','$tfdiero5799','$_110montenegro','@#!(&%5894']

X_predict = vectorizer.transform(X_predict)
Y_Predict = logit.predict(X_predict)
print(Y_Predict)

