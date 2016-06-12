import sys
import os
sys.path.insert(0, os.path.abspath('..'))

from basemodel import BaseModel
import numpy as np
from sklearn.ensemble import BaggingRegressor

class BaggingModel(BaseModel):
    def __init__(self):
        BaseModel.__init__(self)
        return
    def setClf(self):
        self.usedFeatures = [1,2,4,6,7]
#         min_samples_split = 3
        self.clf = BaggingRegressor(n_estimators = 10, max_samples =0.5, max_features =0.5, verbose = 100)
        return
    def getTunedParamterOptions(self):
        tuned_parameters = [{'min_samples_split': np.arange(2, 1000, 1)}]
#         tuned_parameters = [{'min_samples_split': [5, 8,10,12]}]
#         tuned_parameters = [{'min_samples_split': [5, 10]}]
        return tuned_parameters




if __name__ == "__main__":   
    obj= BaggingModel()
    obj.run()