import sys
import os
sys.path.insert(0, os.path.abspath('..'))

from basemodel import BaseModel
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from utility.datafilepath import g_singletonDataFilePath
from preprocess.preparedata import HoldoutSplitMethod

class RandomForestModel(BaseModel):
    def __init__(self):
        BaseModel.__init__(self)
        self.usedFeatures = [101,102,103,4,5,6, 701,702,703,801,802,10,11,1201,1202,1203,1204,1205,1206]
        self.holdout_split = HoldoutSplitMethod.IMITTATE_TEST2_PLUS4
#         self.save_final_model = True
        self.do_cross_val = False
        return
    def get_train_validation_foldid(self):
        return -3
    def setClf(self):
#         min_samples_split = 3
        self.clf = RandomForestRegressor(n_estimators = 800, max_features = 'auto', min_samples_split =1, verbose=100, n_jobs=-1)
        return
    def getTunedParamterOptions(self):
#         tuned_parameters = [{'min_samples_split': np.arange(2, 1000, 1)}]
#         tuned_parameters = [{'min_samples_split': [5, 8,10,12]}]
        tuned_parameters = [{'n_estimators': [2000]}]
        return tuned_parameters




if __name__ == "__main__":   
    obj= RandomForestModel()
    obj.run()