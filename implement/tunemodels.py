import sys
import os
sys.path.insert(0, os.path.abspath('..'))

from implement.decisiontreemodel import DecisionTreeModel
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import ShuffleSplit
from evaluation.sklearnmape import mean_absolute_percentage_error_scoring
from time import time
from utility.datafilepath import g_singletonDataFilePath

class TuneModel:
    def __init__(self):

        return
    def runGridSearch(self, model):
        print "run grid search on model {}".format(model.__class__.__name__)
        
        features,labels = model.getFeaturesLabel()
        # do grid search
        n_iter=2
        estimator = GridSearchCV(model.clf, model.getTunedParamterOptions(), cv=ShuffleSplit(labels.shape[0], n_iter=n_iter,test_size=.25, random_state=10),
                       scoring=mean_absolute_percentage_error_scoring)
        estimator.fit(features, labels)
        model.clf = estimator.best_estimator_
        
        model.dispFeatureImportance()
        print "Best parameters:", estimator.best_params_
        print "Best Scores", -estimator.best_score_ 
        model.predictTestSet(g_singletonDataFilePath.getTest1Dir())
        
        
        return
    def run(self):
        model = DecisionTreeModel()
        model.usedFeatures = [1,4,5,6, 7]
        t0 = time()
        self.runGridSearch(model)
        print "runGridSearch:", round(time()-t0, 3), "s"
        return




if __name__ == "__main__":   
    obj= TuneModel()
    obj.run()