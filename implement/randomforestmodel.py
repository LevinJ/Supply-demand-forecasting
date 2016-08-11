import sys
import os
sys.path.insert(0, os.path.abspath('..'))

from utility.sklearnbasemodel import BaseModel
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from utility.datafilepath import g_singletonDataFilePath
from preprocess.preparedata import HoldoutSplitMethod

class RandomForestModel(BaseModel):
    def __init__(self):
        BaseModel.__init__(self)
#         self.save_final_model = True
        self.do_cross_val = True
        return
    def setClf(self):
#         min_samples_split = 3
#         self.clf = RandomForestRegressor(n_estimators = 100, max_features = 0.3, min_samples_split =1, verbose=100, n_jobs=-1)
        self.clf = RandomForestRegressor(n_estimators = 100, max_features = 0.8)
        return
    def getTunedParamterOptions(self):
#         tuned_parameters = [{'min_samples_split': np.arange(2, 1000, 1)}]
#         tuned_parameters = [{'min_samples_split': [5, 8,10,12]}]
        tuned_parameters = [{'n_estimators': [100], 'max_features':['auto', 0.3,0.8,0.5]}]
        return tuned_parameters




if __name__ == "__main__":   
    obj= RandomForestModel()
    obj.run()