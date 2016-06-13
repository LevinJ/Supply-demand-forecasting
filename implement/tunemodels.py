import sys
import os
sys.path.insert(0, os.path.abspath('..'))

from implement.decisiontreemodel import DecisionTreeModel
from sklearn.grid_search import GridSearchCV
from evaluation.sklearnmape import mean_absolute_percentage_error_scoring
import logging
from utility.logger_tool import Logger
from datetime import datetime
from knnmodel import KNNModel
from utility.duration import Duration
from svmregressionmodel import SVMRegressionModel
from randomforestmodel import RandomForestModel
import numpy as np


class TuneModel:
    def __init__(self):
        self.application_start_time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        logfile_name = r'logs/tunealgorithm_' +self.application_start_time + '.txt'
        _=Logger(filename=logfile_name,filemode='w',level=logging.DEBUG)
        self.durationtool = Duration()
        return
    def runGridSearch(self, model):
        print "run grid search on model {}".format(model.__class__.__name__)
        
        features,labels,cv = model.getFeaturesLabel()
        # do grid search
        estimator = GridSearchCV(model.clf, model.getTunedParamterOptions(), cv=cv,
                       scoring=mean_absolute_percentage_error_scoring, verbose = 500)
        estimator.fit(features, labels)
        model.clf = estimator.best_estimator_
        model.save_final_model = True
        model.save_model()
        
#         model.dispFeatureImportance()
        logging.debug('estimaator parameters: {}'.format(estimator.get_params))
        logging.debug('Best parameters: {}'.format(estimator.best_params_))
        logging.debug('Best Scores: {}'.format(-estimator.best_score_))
        logging.debug('Score grid: {}'.format(estimator.grid_scores_ ))
        for i in estimator.grid_scores_ :
            logging.debug('parameters: {}'.format(i.parameters ))
            logging.debug('mean_validation_score: {}'.format(np.absolute(i.mean_validation_score)))
            logging.debug('cv_validation_scores: {}'.format(np.absolute(i.cv_validation_scores) ))

        
        
        return
    def get_model(self, model_id):
        model_dict = {}
        model_dict[1] =DecisionTreeModel
        model_dict[2] =KNNModel
        model_dict[3] =SVMRegressionModel
        model_dict[4] = RandomForestModel
        return model_dict[model_id]()
    def run(self):
       
        model_id = 4

        model = self.get_model(model_id)
        model.application_start_time = self.application_start_time
        self.durationtool.start()
        self.runGridSearch(model)
        self.durationtool.end()
        return




if __name__ == "__main__":   
    obj= TuneModel()
    obj.run()