import pickle
import pandas as pd
import numpy as np

def Random_forest(payload):
    tree = pickle.load(open('Random_forest.sav','rb'))
    X_test=payload
    prediction=tree.predict(X_test.reshape(1,-1))
    return (int(prediction[0]))

