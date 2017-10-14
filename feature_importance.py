__author__ = 'jliu2188'
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.datasets import load_iris
iris = load_iris()
X, y = iris.data, iris.target

print X.shape
'''clf = ExtraTreesClassifier()
X_new = clf.fit(X,y).transform(X)
print clf.feature_importances_
print X_new.shape'''

# this method is sensitive to correlated feature
new_col = X[:,2] *2
X = np.column_stack((X,new_col))
clf = ExtraTreesClassifier()
X_new = clf.fit(X,y).transform(X)
print clf.feature_importances_
print X_new.shape
