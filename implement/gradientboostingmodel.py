import sys
import os
sys.path.insert(0, os.path.abspath('..'))
import pandas as pd

from basemodel import BaseModel
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from utility.datafilepath import g_singletonDataFilePath
from preprocess.preparedata import HoldoutSplitMethod
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from evaluation.sklearnmape import mean_absolute_percentage_error
import matplotlib.pyplot as plt

class GrientBoostingModel(BaseModel):
    def __init__(self):
        BaseModel.__init__(self)
        self.save_final_model = False
        self.do_cross_val = False
        return
    def setClf(self):
        self.clf = GradientBoostingRegressor(loss = 'ls', learning_rate= 0.02,  n_estimators=10, verbose = 300, subsample=0.5)
        return
    def get_train_validation_foldid(self):
        return 0
    def after_test(self):
        scores_test=[]
        scores_train=[]
        scores_test_mse = []
        scores_train_mse = []
        for i, y_pred in enumerate(self.clf.staged_predict(self.X_test)):
            scores_test.append(mean_absolute_percentage_error(self.y_test, y_pred))
            scores_test_mse.append(mean_absolute_percentage_error(self.y_test, y_pred))
        
        for i, y_pred in enumerate(self.clf.staged_predict(self.X_train)):
            scores_train.append(mean_absolute_percentage_error(self.y_train, y_pred))
            scores_train_mse.append(mean_absolute_percentage_error(self.y_train, y_pred))
        
        pd.DataFrame({'scores_train': scores_train, 'scores_test': scores_test,'scores_train_mse': scores_train_mse, 'scores_test_mse': scores_test_mse}).to_csv('temp/trend.csv')
            
#         pd.DataFrame({'scores_train': scores_train, 'scores_test': scores_test}).plot()
#         plt.show()
#         for i, pred in enumerate(clf.staged_decision_function(X_test)):
#             test_score[i] = clf.loss_(y_test, pred)
#     
#         for i, pred in enumerate(clf.staged_decision_function(X_train)):
#             train_score[i] = clf.loss_(y_train, pred)
#         
#         plot(test_score)
#         plot(train_score)
#         legend(['test score', 'train score'])
#         y_pred_train = self.clf.predict(self.X_train)
#         y_pred_test = self.clf.predict(self.X_test)
#         print "MSE for training set: {}".format(mean_squared_error(self.y_train, y_pred_train))
#         print "MSE for testing set: {}".format(mean_squared_error(self.y_test, y_pred_test))
        return
    def getTunedParamterOptions(self):
        tuned_parameters = [{'n_estimators': [300]}]
#         tuned_parameters = [{'min_samples_split': [5, 8,10,12]}]
#         tuned_parameters = [{'min_samples_split': [5, 10]}]
        return tuned_parameters




if __name__ == "__main__":   
    obj= GrientBoostingModel()
    obj.run()