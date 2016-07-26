import sys
import os
sys.path.insert(0, os.path.abspath('..'))

from basemodel import BaseModel
import numpy as np
from sklearn.svm import SVR
from sklearn import preprocessing
from sklearn.pipeline import Pipeline

class SVMRegressionModel(BaseModel):
    def __init__(self):
        BaseModel.__init__(self)
#         self.usedFeatures = [1,4,5,6,7]
        self.randomSate = None
#         self.excludeZerosActual = True
#         self.test_size = 0.3
        return
    def setClf(self):
        clf = SVR(C=100, epsilon=0.1, gamma = 0.0001,cache_size = 10240)
        min_max_scaler = preprocessing.MinMaxScaler()
        self.clf = Pipeline([('scaler', min_max_scaler), ('estimator', clf)])
        return
    def getTunedParamterOptions(self):
#         tuned_parameters = [{'estimator__C': [0.001,0.01,0.1, 1,10,100,1000, 10000],
#                              'estimator__gamma':[0.00001, 0.0001, 0.001,0.003, 0.01,0.1, 1,10,100,1000,10000]}]
        tuned_parameters = [{'estimator__C': [1,10],
                             'estimator__gamma':[0.00001]}]
        return tuned_parameters




if __name__ == "__main__":   
    obj= SVMRegressionModel()
    obj.run()