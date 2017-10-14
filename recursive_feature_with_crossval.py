__author__ = 'jliu2188'
from sklearn.svm import SVC
import numpy as np
from sklearn.cross_validation import StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.datasets import make_classification

# Build a classification task using 3 informative features
X, y = make_classification(n_samples=1000, n_features=25, n_informative=3,
                           n_redundant=2, n_repeated=0, n_classes=8,
                           n_clusters_per_class=1, random_state=0)

# Create the RFE object and compute a cross-validated score.
svc = SVC(kernel="linear")
# The "accuracy" scoring is proportional to the number of correct
# classifications
rfecv = RFECV(estimator=svc, step=1, cv=StratifiedKFold(y, 2),
              scoring='accuracy')
selector = rfecv.fit(X, y)

print selector.support_
print selector.ranking_
print("Optimal number of features : %d" % rfecv.n_features_)
X_new = selector.transform(X)
print X_new.shape

