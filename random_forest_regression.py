# Random Forest Regression
import numpy as np
from sklearn import datasets
from sklearn.ensemble import RandomForestRegressor
# load the diabetes datasets
dataset = datasets.load_diabetes()
# fit a random forest model to the data
model = RandomForestRegressor()
model.fit(dataset.data, dataset.target)
print(model)
# make predictions
expected = dataset.target
predicted = model.predict(dataset.data)
# summarize the fit of the model
mse = np.mean((predicted-expected)**2)
print(mse)
print(model.score(dataset.data, dataset.target))