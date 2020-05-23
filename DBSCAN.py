import pandas as pd
import numpy as np 
import pickle
from sklearn import preprocessing
import scipy as sp

def dbscan_predict(dbscan_model, X_new, metric=sp.spatial.distance.cosine):
    # Result is noise by default
    y_new = np.ones(shape=len(X_new), dtype=int)*-1 

    # Iterate all input samples for a label
    for j, x_new in enumerate(X_new):
        # Find a core sample closer than EPS
        for i, x_core in enumerate(dbscan_model.components_): 
            if metric(x_new, x_core) < dbscan_model.eps:
                # Assign label of x_core to x_new
                y_new[j] = dbscan_model.labels_[dbscan_model.core_sample_indices_[i]]
                break

    return y_new

def DBSCAN(payload):
    pdf = pd.read_csv("weather_station.csv")
    Clus_dataSet = pdf[['xm','ym','Tx','Tm','Tn']]
    Clus_dataSet = np.nan_to_num(Clus_dataSet)
    scale = preprocessing.StandardScaler().fit(Clus_dataSet)
    Clus_dataSet=scale.transform(Clus_dataSet)
    file = open('DBSCAN.sav', 'rb')
    model = pickle.load(file)
    file.close()
    y=payload
    data=scale.transform(y.reshape(1,-1))
    prediction=dbscan_predict(model,y.reshape(1,-1))
    return int(prediction[0])

