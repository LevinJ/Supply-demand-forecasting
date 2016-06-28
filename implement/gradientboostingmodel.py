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
#         self.save_final_model = True
        self.do_cross_val = False
        return
    def setClf(self):
        self.clf = GradientBoostingRegressor(n_estimators=100, verbose=100)
#         self.clf = GradientBoostingRegressor(loss = 'ls', verbose = 300, n_estimators=70,    learning_rate= 0.1,subsample=1.0, max_features = 1.0)
        return
    def get_train_validation_foldid(self):
        return -1
    def after_test(self):
        scores_test=[]
        scores_train=[]
        scores_test_mse = []
        scores_train_mse = []
        for i, y_pred in enumerate(self.clf.staged_predict(self.X_test)):
            scores_test.append(mean_absolute_percentage_error(self.y_test, y_pred))
            scores_test_mse.append(mean_squared_error(self.y_test, y_pred))
        
        for i, y_pred in enumerate(self.clf.staged_predict(self.X_train)):
            scores_train.append(mean_absolute_percentage_error(self.y_train, y_pred))
            scores_train_mse.append(mean_squared_error(self.y_train, y_pred))
        
        pd.DataFrame({'scores_train': scores_train, 'scores_test': scores_test,'scores_train_mse': scores_train_mse, 'scores_test_mse': scores_test_mse}).to_csv('temp/trend.csv')
        df = pd.DataFrame({'scores_train': scores_train, 'scores_test': scores_test})
        print "Test set MAPE minimum: {}".format(np.array(scores_test).min())
#         df.plot()
#         plt.show()
        return
    def __get_intial_model_param(self):
        
        return {'max_depth': [8],'max_features': [9], 'subsample':[0.8], 'learning_rate':[0.1], 'n_estimators': np.arange(20, 81, 10)}
    def __get_model_param(self):
        return {'max_depth': np.arange(3,15,1),'subsample': np.linspace(0.5, 1.0,6), 'learning_rate':[0.15,0.1,0.08,0.06,0.04,0.02, 0.01], 'n_estimators': [1000,1300,1500,1800,2000]}
    def getTunedParamterOptions(self):
#         tuned_parameters = self.__get_intial_model_param()
        tuned_parameters = self.__get_model_param()
#         tuned_parameters = [{'learning_rate': [0.1,0.05,0.01,0.002],'subsample': [1.0,0.5], 'n_estimators':[15000]}]
#         tuned_parameters = [{'n_estimators': [2]}]
        return tuned_parameters




if __name__ == "__main__":   
    obj= GrientBoostingModel()
    obj.run()