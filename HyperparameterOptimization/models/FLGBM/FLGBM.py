import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.datasets import load_breast_cancer
import os
import contextlib


    
class FairLGBM:
    """
    Class representing the FairLGBM algorithm.
    """
    
    def __init__(self, fair_param, prot, fair_fun, lgbm_params):
        self.fair_param = fair_param
        self.prot = prot
        self.fair_fun = fair_fun
        assert self.fair_fun in ['fpr_diff', 'ppv_diff', 'pnr_diff']
        self.C = None
        lgbm_params['objective'] = self.bce_fair_loss
        self.lgbm_params = lgbm_params
        self.model = None

    def sigmoid(self, x):
        """
        Calculation of a sigmoid value
            Parameters:
                - x: value for which compute the sigmoid function
            Returns:
                - Value of the sigmoid function
        """
        return 1 / (1 + np.exp(-x))


    def _positive_softplus(self, x):
        """
        Auxiliary method for calculating positive softplus function
            Parameters:
                - x: value for which compute the sigmoid
            Returns:
                - Value of the softplus function
        """
        return x + np.log1p(np.exp(-x))

    def _negative_softplus(self, x):
        """
        Auxiliary method for calculating negative softplus function
            Parameters:
                - x: value for which compute the sotfplus function
            Returns:
                - Value of the softplus function
        """
        return np.log1p(np.exp(x))

    def softplus(self, x):
        """
        Calculation of a sotfplus function
            Parameters:
                - x: value for which compute the sotfplus function
            Returns:
                - Value of the softplus function
        """
        positive = x >= 0
        negative = ~positive
        result = np.empty_like(x)
        result[positive] = self._positive_softplus(x[positive])
        result[negative] = self._negative_softplus(x[negative])
        return result


    def bce_fair_loss(self, z, data):
        """
        Custom Binary Cross Entropy with fairness consideration. y is the actual label
        while p is the protected attribute
            Parameters:
                - z : array with predictions
                - data : lightgbm data from which use the information.
            Returns:
                - grad : Gradient of the loss function
                - hess: Hessian of the loss function
        """
        
        grad = 0
        hess = 0
        y = data.get_label()
        s = self.sigmoid(z)
        grad = 1.443 * (1 - self.fair_param) * (s - y) + self.fair_param * self.C * s * (1 - s)
        hess = 1.443 * (1 - self.fair_param) * s * (1 - s) + self.fair_param * self.C * ((s * (1 - s)**2) - (s**2 * (1 - s)))
        return grad, hess
        

    def bce_loss(self, z, data):
        """
        Binary cross entropy loss function
            Parameters:
                - z : array with predictions
                - data : lightgbm data from which use the information.
            Returns:
                - grad : Gradient of the loss function
                - hess: Hessian of the loss function
        """
        y = data.get_label()
        s = self.sigmoid(z)
        grad = s - y
        hess = s * (1 - s)
        return grad, hess


    def bce_eval(self, z, data):
        """
        Binary cross entropy evaluation function
            Parameters:
                - z: Array with predictions
                - data : lightgbm data from which use the information.
            Returns:
                - Metric name, value for the metric, and Boolean value indicating if a higher value is better or not
        """
        t = data.get_label()
        loss = t * self.softplus(-z) + (1 - t) * self.softplus(z)
        return 'bce', loss.mean(), False


    def bce_fair_eval(self, z, data):
        """
        Binary cross entropy with fairness consideration evaluation function
            Parameters:
                - z: Array with predictions
                - data : lightgbm data from which use the information.
            Returns:
                - Metric name, value for the metric, and Boolean value indicating if a higher value is better or not
        """
        y = data.get_label()
        s = self.sigmoid(z)
        loss = - (1-self.fair_param) * (y * np.log(s) + (1 - y) * np.log(1 - s)) + self.fair_param * np.abs(np.dot(self.p_01_val, s) / np.sum(self.p_01_val) - np.dot(self.p_00_val, s) / np.sum(self.p_00_val))
        return 'bce_fair', loss.mean(), False


    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """
        Fit the LightGBM model
            Parameters:
                - X_train: Training data
                - y_train: y attribute to predict
                - X_val: Validation data (optional)
                - y_val: y attribute to validate (optional)
            Returns:
                - a copy of the object, with the trained objective function
        """

        lgb_train = lgb.Dataset(X_train, y_train, free_raw_data=False)
        if self.fair_fun == 'fpr_diff':
            self.p_00 = np.where( (y_train == 0) & (X_train[self.prot] == 0), 1, 0)
            self.p_01 = np.where( (y_train == 0) & (X_train[self.prot] == 1), 1, 0)
            self.C = np.abs(self.p_01 / np.sum(self.p_01) - self.p_00 / np.sum(self.p_00))
            if not X_val is None and not y_val is None:
                self.p_00_val = np.where( (y_val == 0) & (X_val[self.prot] == 0), 1, 0)
                self.p_01_val = np.where( (y_val == 0) & (X_val[self.prot] == 1), 1, 0)

        elif self.fair_fun == 'ppv_diff':
            pass
        elif self.fair_fun == 'pnr_diff':
            pass

        if not X_val is None and not y_val is None:
            lgb_val = lgb.Dataset(data=X_val, label=y_val, reference=lgb_train)
            self.model = lgb.train(self.lgbm_params, lgb_train, valid_sets=[lgb_val], feval=self.bce_fair_eval, callbacks=[lgb.early_stopping(10)])
        else:
            self.model = lgb.train(self.lgbm_params, lgb_train)
        return self


    def predict(self, X_test):
        """
        Predict with trained model the attribute for some new information
            Parameters:
                - X_test: data to predict
            Returns:
                - predictions
        """
        return (self.sigmoid(self.model.predict(X_test)) > 0.5).astype(int)


    def model_to_string(self):
        """
        Converts model to string, for storage purposes
            Returns:
                - model converted to string.
        """
        return self.model.model_to_string()