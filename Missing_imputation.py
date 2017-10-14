__author__ = 'jliu'
import numpy as np
import pandas as pd

def num_missing_mean_median(df, variable, prefix="", mean=True):
    indicator = ""
    if prefix=="":
        indicator = variable+ "_" + "missing"
    else:
        indicator = prefix + "_"+ "missing"
    df[indicator] = np.where(df[variable].isnull(),1,0)
    replaceValue = 0
    if mean== True:
        replaceValue = df[variable].mean()
    else:
        replaceValue = df[variable].median()
    df[variable].fillna(replaceValue, inplace= True)
    return df

# you can use this method or use the dummy coding method defined in dummy_coding
def cat_missing_as_category(df, variable, prefix=""):
    indicator = ""
    if prefix=="":
        indicator = variable+ "_" + "missing"
    else:
        indicator = prefix + "_"+ "missing"
    df[indicator] = np.where(df[variable].isnull(),1,0)
    return df

def cat_missing_mode(df, variable, prefix=""):
    indicator = ""
    if prefix=="":
        indicator = variable+ "_" + "missing"
    else:
        indicator = prefix + "_"+ "missing"
    df[indicator] = np.where(df[variable].isnull(),1,0)
    df[variable].fillna(df[variable].mode().mode[0], inplace=True)
    return df