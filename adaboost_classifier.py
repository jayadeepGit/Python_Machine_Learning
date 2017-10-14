__author__ = 'jliu'
from sklearn.cross_validation import cross_val_score
from sklearn.datasets import load_iris
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
iris = load_iris()
clf = AdaBoostClassifier(n_estimators=100) # decision_tree is the default estimator, n_estimators - how many trees we need to have
scores = cross_val_score(clf, iris.data, iris.target) # by default, we get accuracy for classification and r2 for regression
print scores.mean()

clf = AdaBoostClassifier(SVC(probability=True,kernel='linear'),n_estimators=100)
scores = cross_val_score(clf, iris.data, iris.target)
print scores.mean()
