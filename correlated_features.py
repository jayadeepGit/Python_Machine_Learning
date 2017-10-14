import numpy as np
import sys
from statsmodels.stats.outliers_influence import variance_inflation_factor # you need to install stats models. Please
                                                                           # google statsmodels exe
                                                                           # you also need to install patsy. Please do
                                                                           # easy_install -U patsy
xs = np.random.randn(100, 6)
noise = np.random.randn(100)
xs[:,4] = 2 * xs[:,0] + 2 * xs[:,2] + .5 * noise  # collinearity - one variable is correlated with other variables
corr = np.corrcoef(xs, rowvar=0) # correlaton matrix
print corr
for index in range(xs.shape[1]):
    print variance_inflation_factor(xs, index)
