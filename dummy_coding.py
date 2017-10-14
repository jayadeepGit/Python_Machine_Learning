__author__ = 'jliu'

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn import preprocessing

def dummy_coding(df, variable,  dummy_na=False, drop_first = False, prefix=None):
    if prefix==None:
        prefix = variable
    outputdata = pd.get_dummies(df, columns=[variable], prefix= prefix, dummy_na=dummy_na, drop_first=drop_first)
    return outputdata

def dummy_coding_for_vars(df, list_of_variables,  dummy_na=False, drop_first = False, prefix=None):
    if prefix==None:
        prefix = list_of_variables
    outputdata = pd.get_dummies(df, columns=list_of_variables, prefix= prefix, dummy_na=dummy_na, drop_first=drop_first)
    return outputdata

iris = load_iris()
df= pd.DataFrame(data=np.column_stack((iris.data, iris.target)), columns=["v1", "v2", "v3", "v4", "v5"])
df = dummy_coding(df, "v5", dummy_na=True, drop_first=True) # you set drop_na=True only when are certain this variable contains missing values
print df
