import sys
import os
sys.path.insert(0, os.path.abspath('..'))

from basemodel import BaseModel
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from utility.datafilepath import g_singletonDataFilePath
from preprocess.preparedata import HoldoutSplitMethod

class DecisionTreeModel(BaseModel):
    def __init__(self):
        BaseModel.__init__(self)
        self.save_final_model = False
        return
    def setClf(self):
        min_samples_split = 100
        self.clf = DecisionTreeRegressor(random_state=0, min_samples_split= min_samples_split)
        return
    def getTunedParamterOptions(self):
        tuned_parameters = [{'min_samples_split': np.arange(2, 1000, 1)}]
#         tuned_parameters = [{'min_samples_split': [5, 8,10,12]}]
#         tuned_parameters = [{'min_samples_split': [5, 10]}]
        return tuned_parameters




if __name__ == "__main__":   
    obj= DecisionTreeModel()
    obj.run()