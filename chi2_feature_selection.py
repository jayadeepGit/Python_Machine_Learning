__author__ = 'jliu2188'
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
iris = load_iris()
X, y = iris.data, iris.target
print X.shape
chi2_values, p_values = chi2(X, y)
print chi2_values
for p_value in p_values:
    print("%.4f" % p_value)
X_new = SelectKBest(chi2, k=2).fit_transform(X, y)
print X_new.shape
