# SVM Classification
from sklearn import datasets
from sklearn import metrics
from sklearn.svm import SVC
# load the iris datasets
from sklearn.datasets import make_classification

# Build a classification task using 3 informative features
X, y = make_classification(n_samples=1000, n_features=25, n_informative=3,
                           n_redundant=2, n_repeated=0, n_classes=8,
                           n_clusters_per_class=1, random_state=0)
# fit a SVM model to the data
model = SVC(kernel = 'rbf')
model.fit(X, y)
print(model)
# make predictions

predicted = model.predict(X)
# summarize the fit of the model
print(metrics.classification_report(y, predicted))
print(metrics.confusion_matrix(y, predicted))