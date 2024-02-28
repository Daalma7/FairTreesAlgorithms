import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.datasets import load_breast_cancer
import os
import contextlib


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
    
    def __init__(self, lamb, proc, fair_c, lgbm_params):
        self.lamb = lamb
        self.proc = proc
        self.fair_c = fair_c
        #lgb.register_obj(self.bce_fair_loss)
        #lgbm_params['objective'] = self.bce_fair_loss
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
        
        grad = 0
        hess = 0
        y = data.get_label()
        p = data.get_data()[self.proc]
        t = self.sigmoid(z)

        if self.fair_c == 'fpr':
            s01 = (np.logical_and(y==1, p==1)).astype(int)
            s00 = (np.logical_and(y==1, p==0)).astype(int)
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
        
        if self.fair_c == 'ppv':
            pass

        if self.fair_c == 'pnr':
            pass
        return grad, hess
        

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


    def bce_fair_eval(self, z, data):
        y = data.get_label()
        loss = y * self.softplus(-z) + (1 - y) * (self.softplus(z) - np.abs(np.dot(sum_s00 * s01 - sum_s01 * s00, self.softplus(z)) / (sum_s01 * sum_s00)))
        return 'bce', loss.mean(), False


    def fit(self, X_train, y_train, X_val=None, y_val=None):
        lgb_train = lgb.Dataset(X_train, y_train, free_raw_data=False)
        #with open(os.devnull, 'w') as f, contextlib.redirect_stdout(f):
        #self.model = lgb.train(self.lgbm_params, lgb_train, feval=self.bce_eval)
        if not X_val is None and not y_val is None:
            self.model = lgb.train(self.lgbm_params, lgb_train, eval_set=[X_val, y_val])
        else:
            self.model = lgb.train(self.lgbm_params, lgb_train)
        return self


    def predict(self, X_test):
        #print(self.model.predict(X_test))
        #print(self.sigmoid(self.model.predict(X_test)))
        #print((self.sigmoid(self.model.predict(X_test)) > 0.5).astype(int))
        return (self.sigmoid(self.model.predict(X_test)) > 0.5).astype(int)


    def model_to_string(self):
        return self.model.model_to_string()