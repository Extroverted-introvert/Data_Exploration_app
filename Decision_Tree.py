import pickle
import pandas as pd
import numpy as np

def Decision_Tree(payload):
    tree = pickle.load(open('decision_tree_model.sav','rb'))
    X_test=payload
    prediction=tree.predict(X_test.reshape(1,-1))
    return prediction[0]
 