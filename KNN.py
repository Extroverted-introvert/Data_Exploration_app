import pandas as pd
import numpy as np 
import pickle
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier

def KNN(payload):
    df = pd.read_csv('teleCust1000t.csv')
    X = df[['region', 'tenure','age', 'marital', 'address', 'income', 'ed', 'employ','retire', 'gender', 'reside']] .values  #.astype(float)
    scale = preprocessing.StandardScaler().fit(X)
    X=scale.transform(X)
    y=df['custcat'].values
    model =  KNeighborsClassifier(n_neighbors = 9).fit(X,y)
    payload=scale.transform(payload.reshape(1,-1))
    prediction=model.predict(payload.reshape(1,-1))
    return int(prediction[0])