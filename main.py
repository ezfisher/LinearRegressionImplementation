from load_data import x_train, x_test, y_train, y_test
from multivariate_linear_regression import MultivariateLinearRegression
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd

'''
Comparing the performance of my linear regression model to that of sklearn
'''
scores = []

my_clf = MultivariateLinearRegression()
my_clf.fit(x_train, y_train)
my_pred = my_clf.predict(x_test)
scores.append(my_clf.r_squared(y_test, my_pred))

clf = LinearRegression()
clf.fit(x_train, y_train)
pred = clf.predict(x_test)
scores.append(clf.score(x_test, y_test))

print('multivariate regression performance: ',scores[0],'\nsklearn regression performance: ',scores[1])