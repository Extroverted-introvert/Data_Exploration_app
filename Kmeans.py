import pandas as pd
import numpy as np 
import pickle
from sklearn import preprocessing

def Kmeans(payload):
    cust_df = pd.read_csv("Cust_Segmentation.csv")
    df = cust_df.drop('Address', axis=1)
    X = df.values[:,1:]
    X = np.nan_to_num(X)
    scale = preprocessing.StandardScaler().fit(X)    
    file = open('Kmeans.sav', 'rb')
    model = pickle.load(file)
    file.close()
    centroids=model.cluster_centers_
    centroids = centroids[::-1] 
    new_data = payload
    new_data=scale.transform(new_data.reshape(1,-1))
    diff = centroids - new_data  
    dist = np.sqrt(np.sum(diff**2, axis=-1)) 
    closest_centroid = np.argmin(dist)
    #print(closest_centroid)
    return int(closest_centroid)