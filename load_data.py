'''
Testing liner regression performance on sklearn's Boston housing data
'''

from sklearn.datasets import load_boston
import pandas as pd
from sklearn.model_selection import train_test_split


boston_data = load_boston()
boston = pd.DataFrame(boston_data.data, columns=boston_data.feature_names)
boston['MEDV'] = boston_data.target

x = boston.drop(['MEDV'], axis=1)
y = boston['MEDV']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=48)