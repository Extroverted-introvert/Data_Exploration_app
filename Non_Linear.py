import pandas as pd
import numpy as np
import pickle

def sigmoid(x, Beta_1, Beta_2):
     y = 1 / (1 + np.exp(-Beta_1*(x-Beta_2)))
     return y

def Non_Linear_Regression(year):
    year=year/2014
    model=pickle.load(open('non_linear_finalized_model.sav','rb'))
    prediction=sigmoid(year,*model[0])
    return prediction*10354831729340.4


