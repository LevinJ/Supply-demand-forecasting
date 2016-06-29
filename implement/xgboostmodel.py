import sys
import os
sys.path.insert(0, os.path.abspath('..'))

from basemodel import BaseModel
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from utility.datafilepath import g_singletonDataFilePath
from preprocess.splittrainvalidation import HoldoutSplitMethod
import matplotlib.pyplot as plt
import xgboost as xgb
from evaluation.sklearnmape import mean_absolute_percentage_error_xgboost
from evaluation.sklearnmape import mean_absolute_percentage_error
from utility.xgboostgridsearch import XGBoost_GridSearch

class XGBoostModel(BaseModel):
    def __init__(self):
        BaseModel.__init__(self)
#         self.save_final_model = True
#         self.do_cross_val = False
        return
    def run_croos_validation(self, dtrain,cv):
         
        # specify parameters via map, definition are same as c++ version
        param = {'max_depth':14, 'eta':0.02, 'silent':1, 'objective':'reg:linear' }
         
        # specify validations set to watch performance
        num_boost_round = 50
        early_stopping_rounds = 3
        bst = xgb.cv(param, dtrain, num_boost_round=num_boost_round,  feval = mean_absolute_percentage_error_xgboost, folds = cv,callbacks=[xgb.callback.print_evaluation(show_stdv=True),
                        xgb.callback.early_stop(early_stopping_rounds)])
         
#         print bst

        return 
    def run_train_validation(self):
        fold_id = -1
        self.get_train_validationset(fold_id)
        
        dtrain = xgb.DMatrix(self.X_train, label= self.y_train,feature_names=self.X_train.columns)
        dtest =  xgb.DMatrix(self.X_test, label= self.y_test,feature_names=self.X_test.columns)
         
        # specify parameters via map, definition are same as c++ version
        param = {'max_depth':14, 'eta':0.02, 'silent':1, 'objective':'reg:linear' }
         
        # specify validations set to watch performance
        evals  = [(dtrain,'train'),(dtest,'eval')]
        num_boost_round = 50
        early_stopping_rounds=3
        bst = xgb.train(param, dtrain, num_boost_round=num_boost_round, evals = evals, early_stopping_rounds=early_stopping_rounds,feval = mean_absolute_percentage_error_xgboost)
         
 
        y_pred_train = bst.predict(dtrain)
        y_pred_test = bst.predict(dtest)
        print "features used:\n {}".format(self.usedFeatures)
         
        print "MAPE for training set: {}".format(mean_absolute_percentage_error(self.y_train, y_pred_train))
        print "MAPE for testing set: {}".format(mean_absolute_percentage_error(self.y_test, y_pred_test))
        return
    def run_grid_search(self, dtrain,cv):
#         param_grid = {'max_depth':[7,8], 'eta':[0.01], 'silent':[1], 'objective':['reg:linear'] }
        param_grid = {'max_depth':range(5,15), 'eta':[0.01, 0.02,0.005], 'silent':[1], 'objective':['reg:linear'] }
        num_boost_round = 100
        early_stopping_rounds = 3
        grid = XGBoost_GridSearch( param_grid,  cv=cv, num_boost_round = num_boost_round, 
                                   early_stopping_rounds=early_stopping_rounds,feval = mean_absolute_percentage_error_xgboost)
        grid.fit(dtrain)
        return
    def run(self):
        sel = 3
        if sel == 1:
            return self.run_train_validation()
        
        # now for cross validation and parameter tuning
        features,labels,cv = self.getFeaturesLabel()
        dtrain = xgb.DMatrix(features, label= labels,feature_names=features.columns)
        if sel == 2:
            return self.run_croos_validation(dtrain,cv)
        self.run_grid_search(dtrain,cv)     
        
         
        return




if __name__ == "__main__":   
    obj= XGBoostModel()
    obj.run()