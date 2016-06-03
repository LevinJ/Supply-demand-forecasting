from basemodel import BaseModel
import numpy as np
from sklearn.tree import DecisionTreeRegressor

class DecisionTreeModel(BaseModel):
    def __init__(self):
        BaseModel.__init__(self)
#         self.usedFeatures = ['gap1', 'gap2', 'gap3']
        return
    def setClf(self):
        self.clf = DecisionTreeRegressor(random_state=0)
        return
    def dispFeatureImportance(self, clf):
        if not hasattr(clf, 'feature_importances_'):
            return
        features_list = np.asanyarray(self.usedFeatures)
        sortIndexes = clf.feature_importances_.argsort()[::-1]
        features_rank = features_list[sortIndexes]
        num_rank = clf.feature_importances_[sortIndexes]
        print "Ranked features: {}".format(features_rank)
        print "Ranked importance: {}".format(num_rank)
        return
    def afterTrain(self):
        self.dispFeatureImportance(self.clf)
        return




if __name__ == "__main__":   
    obj= DecisionTreeModel()
    obj.run()