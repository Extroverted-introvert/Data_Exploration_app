import pickle
import pandas as pd
import numpy as np
from sklearn import svm

def SVM_model(payload):
    cell_df = pd.read_csv("cell_samples.csv")
    cell_df = cell_df[pd.to_numeric(cell_df['BareNuc'], errors='coerce').notnull()]
    cell_df['BareNuc'] = cell_df['BareNuc'].astype('int')
    feature_df = cell_df[['Clump', 'UnifSize', 'UnifShape', 'MargAdh', 'SingEpiSize', 'BareNuc', 'BlandChrom', 'NormNucl', 'Mit']]
    X = np.asarray(feature_df)
    cell_df['Class'] = cell_df['Class'].astype('int')
    y = np.asarray(cell_df['Class'])
    clf1 = svm.SVC(kernel='rbf')
    model=clf1.fit(X, y)
    X_test=payload
    prediction=(model.predict(X_test.reshape(1,-1)))
    return (int(prediction[0]))
