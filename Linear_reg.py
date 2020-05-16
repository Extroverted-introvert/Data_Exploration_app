import pandas as pd
import numpy as np
from sklearn import linear_model
import pickle

def Linear_Regression(engine):
    regr = pickle.load(open('finalized_model.sav', 'rb'))
    # The coefficients
    #engine=int(engine)
    #print(type(regr.coef_[0][0]),type(regr.intercept_[0]),type(engine))
    prediction=regr.coef_[0][0]*engine + regr.intercept_[0]
    #print (prediction)
    return prediction

#print(Linear_Regression(3))