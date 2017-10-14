__author__ = 'jliu'
from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris

iris = load_iris()

clf = RandomForestClassifier(n_estimators=10)
scores = cross_val_score(clf, iris.data, iris.target)
print scores.mean()

clf = RandomForestClassifier(n_estimators=100)
scores = cross_val_score(clf, iris.data, iris.target)
print scores.mean()

clf = ExtraTreesClassifier(n_estimators=100)
scores = cross_val_score(clf, iris.data, iris.target)
print scores.mean()

