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

class XGBoostModel(BaseModel):
    def __init__(self):
        BaseModel.__init__(self)
#         self.save_final_model = True
        self.do_cross_val = False
        return
    def get_train_validation_foldid(self):
        return -3
    def run_croos_validation(self):
        features,labels,cv = self.getFeaturesLabel()
        
        dtrain = xgb.DMatrix(features, label= labels,feature_names=features.columns)
         
        # specify parameters via map, definition are same as c++ version
        param = {'max_depth':7, 'eta':0.01, 'silent':1, 'objective':'reg:linear' }
         
        # specify validations set to watch performance
        num_round = 100
        early_stopping_rounds = 3
        bst = xgb.cv(param, dtrain, num_boost_round=num_round,  feval = mean_absolute_percentage_error_xgboost, folds = cv,callbacks=[xgb.callback.print_evaluation(show_stdv=True),
                        xgb.callback.early_stop(early_stopping_rounds)])
         
        print bst

        return 
    def run_tainvalidation(self):
        self.get_train_validationset(foldid= self.get_train_validation_foldid())
        dtrain = xgb.DMatrix(self.X_train, label= self.y_train,feature_names=self.X_train.columns)
        dtest =  xgb.DMatrix(self.X_test, label= self.y_test,feature_names=self.X_test.columns)
         
        # specify parameters via map, definition are same as c++ version
        param = {'max_depth':10, 'eta':0.01, 'silent':1, 'objective':'reg:linear' }
         
        # specify validations set to watch performance
        watchlist  = [(dtrain,'train'),(dtest,'eval')]
        num_round = 100
        bst = xgb.train(param, dtrain, num_boost_round=num_round, evals = watchlist, feval = mean_absolute_percentage_error_xgboost, early_stopping_rounds=3)
         
 
        y_pred_train = bst.predict(dtrain)
        y_pred_test = bst.predict(dtest)
        print "features used:\n {}".format(self.usedFeatures)
         
        print "MAPE for training set: {}".format(mean_absolute_percentage_error(self.y_train, y_pred_train))
        print "MAPE for testing set: {}".format(mean_absolute_percentage_error(self.y_test, y_pred_test))
        return
    def run(self):
#         self.run_tainvalidation()
        self.run_croos_validation()
         
        return




if __name__ == "__main__":   
    obj= XGBoostModel()
    obj.run()