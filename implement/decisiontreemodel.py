import sys
import os
sys.path.insert(0, os.path.abspath('..'))

from basemodel import BaseModel
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from utility.datafilepath import g_singletonDataFilePath

class DecisionTreeModel(BaseModel):
    def __init__(self):
        BaseModel.__init__(self)
        self.usedFeatures = [1,4,5,6,7]
        self.randomSate = None
#         self.excludeZerosActual = True
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
    def dispFeatureImportance(self):
        if not hasattr(self.clf, 'feature_importances_'):
            return
        features_list = np.asanyarray(self.usedFeatures)
        sortIndexes = self.clf.feature_importances_.argsort()[::-1]
        features_rank = features_list[sortIndexes]
        num_rank = self.clf.feature_importances_[sortIndexes]
        print "Ranked features: {}".format(features_rank)
        print "Ranked importance: {}".format(num_rank)
        return
    def afterTrain(self):
        self.dispFeatureImportance()
        return
#     def afterRun(self):
#         self.predictTestSet(g_singletonDataFilePath.getTest1Dir())
#         return




if __name__ == "__main__":   
    obj= DecisionTreeModel()
    obj.run()