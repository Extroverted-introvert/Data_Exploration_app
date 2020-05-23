from sklearn import decomposition
import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.preprocessing import scale


def PCA(payload):
    cell_df = pd.read_csv("cell_samples.csv")
    cell_df = cell_df[pd.to_numeric(cell_df['BareNuc'], errors='coerce').notnull()]
    cell_df['BareNuc'] = cell_df['BareNuc'].astype('int')
    feature_df = cell_df[['Clump', 'UnifSize', 'UnifShape', 'MargAdh', 'SingEpiSize', 'BareNuc', 'BlandChrom', 'NormNucl', 'Mit']]
    X = np.asarray(feature_df)
    X=scale(X)
    cell_df['Class'] = cell_df['Class'].astype('int')
    y = np.asarray(cell_df['Class'])
    pca = decomposition.PCA(n_components=3)
    pca.fit(X)
    X = pca.transform(X)
    clf1 = svm.SVC(kernel='linear')
    model=clf1.fit(X, y)
    X_test=payload
    X_test = pca.transform(X_test.reshape(1,-1))
    prediction=(model.predict(X_test.reshape(1,-1)))
    return (int(prediction[0]))