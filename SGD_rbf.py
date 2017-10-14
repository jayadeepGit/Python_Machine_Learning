__author__ = 'jliu'

from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import SGDClassifier
from sklearn.datasets import make_classification
from sklearn import metrics

# Build a classification task using 3 informative features
X, y = make_classification(n_samples=1000, n_features=25, n_informative=3,
                           n_redundant=2, n_repeated=0, n_classes=8,
                           n_clusters_per_class=1, random_state=0)
from sklearn.preprocessing import StandardScaler
# in order to use RBF kernel, we need to standardize the training dataset. Actually, I would suggest to standardize data for all SVM algorithms
scaler = StandardScaler()
X = scaler.fit_transform(X)
rbf_feature = RBFSampler(gamma=1, random_state=1)
X_features = rbf_feature.fit_transform(X)
clf = SGDClassifier(alpha=1)
clf.fit(X_features, y)

# make predictions
predicted = clf.predict(X_features)
# summarize the fit of the model
print(metrics.classification_report(y, predicted))
print(metrics.confusion_matrix(y, predicted))