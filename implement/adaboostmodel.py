import sys
import os
sys.path.insert(0, os.path.abspath('..'))

from utility.sklearnbasemodel import BaseModel
import numpy as np
from sklearn.ensemble import AdaBoostRegressor

class AdaBoostModel(BaseModel):
    def __init__(self):
        BaseModel.__init__(self)
        self.usedFeatures = [1,4,5,6,7]
        self.randomSate = None
        self.excludeZerosActual = True
        return
    def setClf(self):
#         min_samples_split = 3
        self.clf = AdaBoostRegressor()
        return
    def getTunedParamterOptions(self):
        tuned_parameters = [{'min_samples_split': np.arange(2, 1000, 1)}]
#         tuned_parameters = [{'min_samples_split': [5, 8,10,12]}]
#         tuned_parameters = [{'min_samples_split': [5, 10]}]
        return tuned_parameters




if __name__ == "__main__":   
    obj= AdaBoostModel()
    obj.run()