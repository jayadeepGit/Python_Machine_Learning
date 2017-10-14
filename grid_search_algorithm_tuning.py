import numpy as np
from sklearn.linear_model import Lasso
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

boston = load_boston()
X = boston.data
y = boston.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.4,
                                                    random_state=0)

# select one parameter alpha for lasso. Uncomment it to take a look
parameters = np.arange(1,11)/100.0
parameter_grid = [{'alpha': parameters.tolist()}]
lasso = Lasso()
model = GridSearchCV(lasso, parameter_grid, cv = 10, scoring = 'r2')
model.fit(X_train, y_train)
print "Best parameters set found on development set:"
print model.best_estimator_
y_pred_lasso= model.predict(X_test)
print "r^2 on test data : %f" % (1 - np.linalg.norm(y_test - y_pred_lasso) ** 2/np.linalg.norm(y_test) ** 2)
lasso = Lasso(alpha=0.4)
y_pred_lasso=lasso.fit(X_train, y_train).predict(X_test)
print "r^2 on test data : %f" % (1 - np.linalg.norm(y_test - y_pred_lasso) ** 2/np.linalg.norm(y_test) ** 2)

# # select two parameters (alpha and L1_ratio) for ElasticNet
# from sklearn.linear_model import ElasticNet
# enet = ElasticNet()
# alphas = np.arange(1,11)/100.0
# l1_ratios = np.arange(1,11)/100.0
# print alphas
# parameter_grid = [{'alpha': alphas.tolist(), 'l1_ratio':l1_ratios}]
# model = GridSearchCV(enet, parameter_grid, cv = 10, scoring = 'r2')
# model.fit(X_train, y_train)
# print "Best parameters set found on development set:"
# print model.best_estimator_
# y_pred_enet = model.fit(X_train, y_train).predict(X_test)
# print "r^2 on test data : %f" % (1 - np.linalg.norm(y_test - y_pred_enet) ** 2/np.linalg.norm(y_test) ** 2)

