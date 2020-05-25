import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.naive_bayes import GaussianNB
import pickle


def Naive_Bayes(payload):
    churn_df = pd.read_csv("ChurnData.csv")
    X = np.asarray(churn_df[['tenure', 'age', 'address', 'income', 'ed', 'employ']])
    model = pickle.load(open('Naive_Bayes.sav','rb'))
    X_test=payload
    prediction=model.predict(X_test.reshape(1,-1))
    return int(prediction[0])
