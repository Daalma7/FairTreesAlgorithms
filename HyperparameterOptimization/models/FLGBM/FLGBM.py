import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.datasets import load_breast_cancer
import seaborn as sns


def _positive_sigmoid(x):
    return 1 / (1 + np.exp(-x))

def _negative_sigmoid(x):
    exp = np.exp(x)
    return exp / (exp + 1)

def sigmoid(x):
    positive = x >= 0
    negative = ~positive
    result = np.empty_like(x)
    result[positive] = _positive_sigmoid(x[positive])
    result[negative] = _negative_sigmoid(x[negative])
    return result


class FairLGBM:
    
    def __init__(self, lamb, proc, lgbm_params):
        self.lamb = lamb
        self.proc = proc
        self.lgbm_params = lgbm_params
        self.model = None

    def _positive_sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def _negative_sigmoid(self, x):
        exp = np.exp(x)
        return exp / (exp + 1)
    
    def sigmoid(self, x):
        positive = x >= 0
        negative = ~positive
        result = np.empty_like(x)
        result[positive] = self._positive_sigmoid(x[positive])
        result[negative] = self._negative_sigmoid(x[negative])
        return result
    
    
    def _positive_softplus(self, x):
        return x + np.log1p(np.exp(-x))
    
    def _negative_softplus(self, x):
        return np.log1p(np.exp(x))
    
    def softplus(self, x):
        positive = x >= 0
        negative = ~positive
        result = np.empty_like(x)
        result[positive] = self._positive_softplus(x[positive])
        result[negative] = self._negative_softplus(x[negative])
        return result
    
    
    def bce_loss(self, z, data):
        t = data.get_label()
        y = self.sigmoid(z)
        grad = y - t
        hess = y * (1 - y)
        return grad, hess


    def bce_eval(self, z, data):
        t = data.get_label()
        loss = t * self.softplus(-z) + (1 - t) * self.softplus(z)
        return 'bce', loss.mean(), False

        
    def bce_fair_loss(self, z, data):
        """
        Parameters
        ----------
        z : array
            predictions
        data : lightgbm data
            lightgbm data from which use the information.
    
        Returns
        -------
        grad : TYPE
            DESCRIPTION.
        hess : TYPE
            DESCRIPTION.
            
        Additional info
        -------
        y is the actual label
        p is the protected attribute
        
    
        """
        y = data.get_label()
        p = data.get_data()[self.proc]
        t = self.sigmoid(z)
    
        s01 = (np.logical_and(y==0, p==1)).astype(int)
        s00 = (np.logical_and(y==0, p==0)).astype(int)
        sum_s01 = np.sum(s01)
        sum_s00 = np.sum(s00)
        fracdiff = np.dot(sum_s00 * s01 - sum_s01 * s00, t) / (sum_s01 * sum_s00)
        
        
        if not fracdiff == 0:
            lastpart = (s01 / sum_s01 - s00 / sum_s00) * np.sign(fracdiff) 
            grad = self.lamb * (t - y) + (1 - self.lamb) * (1 - y) * t * (1 - t) * lastpart
            hess = self.lamb * t * (1 - t) + (1 - self.lamb) * (1 - y) * t * (1 - t) * (1 - 2*t) * lastpart
        else:
            grad = (t - y)
            hess = t * (1 - t)
        return grad, hess
    
    
    def bce_fair_eval(self, z, data):
        y = data.get_label()
        loss = y * self.softplus(-z) + (1 - y) * (self.softplus(z) - np.abs(np.dot(sum_s00 * s01 - sum_s01 * s00, self.softplus(z)) / (sum_s01 * sum_s00)))
        return 'bce', loss.mean(), False
    
    def fit(self, X_train, y_train):
        lgb_train = lgb.Dataset(X_train, y_train, free_raw_data=False)
        self.model = lgb.train(self.lgbm_params, lgb_train, fobj=self.bce_fair_loss, feval=self.bce_eval)
        return self

    def predict(self, X_test):
        #print(self.model.predict(X_test))
        #print(self.sigmoid(self.model.predict(X_test)))
        #print((self.sigmoid(self.model.predict(X_test)) > 0.5).astype(int))
        return (self.sigmoid(self.model.predict(X_test)) > 0.5).astype(int)
    
    def model_to_string(self):
        return self.model.model_to_string()
