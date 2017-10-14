__author__ = 'jliu2188'
# Recursive Feature Elimination
from sklearn import datasets
from sklearn.feature_selection import RFE
from sklearn.svm import SVC
import numpy as np
# load the iris datasets
dataset = datasets.load_iris()
X = dataset.data
y = dataset.target
# create a base classifier used to evaluate a subset of attributes
svc = SVC(kernel="linear", C=1)
# create the RFE model and select 2 attributes
rfe = RFE(estimator=svc, n_features_to_select=2, step=1)
rfe.fit(X, y)
# summarize the selection of the attributes
print(rfe.support_)
print(rfe.ranking_)