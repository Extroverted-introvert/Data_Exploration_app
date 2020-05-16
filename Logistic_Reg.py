import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
import pickle


def Logistic_Regression(payload):
    churn_df = pd.read_csv("ChurnData.csv")
    X = np.asarray(churn_df[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip']])
    X_model = preprocessing.StandardScaler().fit(X)
    LR = pickle.load(open('logistic_finalized_model.sav','rb'))
    X_test=payload
    X_test=X_model.transform(X_test.reshape(1,-1))
    prediction=LR.predict(X_test)
    return str(prediction[0])

