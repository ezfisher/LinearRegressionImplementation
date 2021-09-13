'''
My implementation of linear regression
'''

import numpy as np
import copy as cp
#import copy


class MultivariateLinearRegression():

    def __init__(self):
        self.coefficients = None
        self.intercept = None
    
    def fit(self, x, y):
        # function that will fit the training data
        x = self._transform_x(x)
        y = self._transform_y(y)
        beta = self._estimate_coefficients(x, y)

        # intercept is the 0th component of the beta vector
        self.intercept = beta[0]
        # the coefficients are the rest of the beta vector
        self.coefficients = beta[1:]

    def predict(self, x):

        '''
        y = beta_0*x_0 + beta_1*x_1 + ...
        the beta_0*x_0 term is actually the intercept, which shouldn't be multiplied by x
        need to insert 1s at the 0th component of x to account for this
        '''
        predictions = []

        for index, row in x.iterrows():
            values = row.values
            pred = np.multiply(values, self.coefficients)
            pred = sum(pred)
            pred += self.intercept
            predictions.append(pred)

        return predictions
    
    def r_squared(self, y_true, y_pred):
        '''
        r-squared for evaluating the model
        r2 = 1 - (sum of squared residual errors)/(sum of squared total errors)
        '''

        y_values = y_true.values
        y_avg = np.mean(y_values)

        residual_sum_squares = 0
        total_sum_squares = 0

        for i in range(len(y_values)):
            residual_sum_squares += (y_values[i] - y_pred[i])**2
            total_sum_squares += (y_values[i] - y_avg)**2
        
        return 1 - residual_sum_squares/total_sum_squares
        
 
    def _transform_x(self, x):
        x = cp.deepcopy(x)
        x.insert(0, 'ones', np.ones( (x.shape[0], 1) ) )
        return x.values

    def _transform_y(self, y):
        y = cp.deepcopy(y)
        return y.values
    
    def _estimate_coefficients(self, x, y):
        '''
        estimate beta
        beta = (x^T * x)^-1 * x^T * y
        '''

        xT = np.transpose(x)
        inverse = np.linalg.inv(xT.dot(x))

        coefficients = inverse.dot(xT).dot(y)
        return coefficients
