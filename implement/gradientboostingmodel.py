import sys
import os
sys.path.insert(0, os.path.abspath('..'))

from basemodel import BaseModel
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from utility.datafilepath import g_singletonDataFilePath
from preprocess.preparedata import HoldoutSplitMethod
from sklearn.ensemble import GradientBoostingRegressor

class GrientBoostingModel(BaseModel):
    def __init__(self):
        BaseModel.__init__(self)
        self.save_final_model = False
        self.do_cross_val = False
        return
    def setClf(self):
        self.clf = GradientBoostingRegressor(verbose = 300)
        return
    def get_train_validation_foldid(self):
        return -3
    def afterTrain(self):
        self.dispFeatureImportance()
        return
    def getTunedParamterOptions(self):
        tuned_parameters = [{'min_samples_split': np.arange(2, 10, 1)}]
#         tuned_parameters = [{'min_samples_split': [5, 8,10,12]}]
#         tuned_parameters = [{'min_samples_split': [5, 10]}]
        return tuned_parameters




if __name__ == "__main__":   
    obj= GrientBoostingModel()
    obj.run()