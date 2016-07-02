import sys
import os
sys.path.insert(0, os.path.abspath('..'))

from preprocess.preparedata import PrepareData
import numpy as np
from utility.runtype import RunType
from utility.datafilepath import g_singletonDataFilePath
from preprocess.splittrainvalidation import HoldoutSplitMethod
import matplotlib.pyplot as plt
import xgboost as xgb
from evaluation.sklearnmape import mean_absolute_percentage_error_xgboost
from evaluation.sklearnmape import mean_absolute_percentage_error
# from utility.xgboostgridsearch import GridSearchCV
# from utility.xgboostgridsearch import RandomizedSearchCV
from utility.gridsearchexgboost import GridSearchXGBoost



    
class XGBoostModel(GridSearchXGBoost):
    def __init__(self):
        GridSearchXGBoost.__init__(self)
        self.dtrain = None
        self.folds_params = None 
        self.run_type = RunType.RUN_GRID_SEARCH
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
        fold_id = -3
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
#     def adjust_cv_param(self):
#         self.ramdonized_search_enable = False
#         self.randomized_search_n_iter = 3
#         
#         
#         num_boost_round = 3
#         early_stopping_rounds = 3
#         folds = 5
#         
#         kwargs = {'num_boost_round':num_boost_round, 'folds':folds,
#                   'callbacks':[xgb.callback.print_evaluation(show_stdv=True),xgb.callback.early_stop(early_stopping_rounds)]}
#         return kwargs
    def get_model_input(self):
        """
        self.dtrain, the data to be trained
        self.folds_params, cross validation folds index, and evaluation metric
        self.run_type, how to run the model
        """
        self.dtrain = None
        self.folds_params = None 
        self.run_type = RunType.RUN_GRID_SEARCH
        return
    def run(self):
        run_dict = {}
        run_dict[RunType.RUN_TRAIN_VALIDATION] = self.run_train_validation
        run_dict[RunType.RUN_GRID_SEARCH] = self.run_grid_search
        run_dict[RunType.RUN_CROSS_VALIDATION] = self.run_croos_validation
       
        self.get_model_input()
        run_dict[self.run_type]()        
         
        return

class DidiXGBoostModel(XGBoostModel, PrepareData):
    def __init__(self):
        PrepareData.__init__(self)
        XGBoostModel.__init__(self)
#         self.holdout_split = HoldoutSplitMethod.IMITTATE_TEST2_MIN #[78]    train-mape:-0.373462+0.00492894    test-mape:-0.46214+0.0114662
#         self.holdout_split = HoldoutSplitMethod.IMITTATE_TEST2_PLUS1 # [79]    train-mape:-0.396072+0.00363566    test-mape:-0.459982+0.0100845
#         self.holdout_split = HoldoutSplitMethod.IMITTATE_TEST2_PLUS3 #[78]    train-mape:-0.411597+0.00219856    test-mape:-0.454906+0.0124385
#         self.holdout_split = HoldoutSplitMethod.IMITTATE_TEST2_PLUS4#[78]    train-mape:-0.411597+0.00219856    test-mape:-0.454906+0.0124385
#         self.holdout_split = HoldoutSplitMethod.IMITTATE_TEST2_PLUS6# [82]    train-mape:-0.421504+0.00191357    test-mape:-0.453868+0.0116971
#         self.holdout_split = HoldoutSplitMethod.IMITTATE_TEST2_FULL#[81]    train-mape:-0.428109+0.000703088    test-mape:-0.451429+0.0114696
        return
    def get_model_input(self):
        features,labels,cv = self.getFeaturesLabel()
        dtrain = xgb.DMatrix(features, label= labels,feature_names=features.columns)
        
        self.dtrain = dtrain
        self.folds_params = {'folds':cv, 'feval':mean_absolute_percentage_error_xgboost}
        self.run_type = RunType.RUN_GRID_SEARCH
        return
    def adjust_intial_param(self):
        param = {'max_depth':6, 'eta':0.1,  'min_child_weight':1,'silent':1, 
                 'objective':'reg:linear','colsample_bytree':0.8,'subsample':0.8,'min_child_weight':1 }
        return param
    
    def adjust_param(self, param_grid):
        self.ramdonized_search_enable = False
        self.randomized_search_n_iter = 2
        self.grid_search_display_result = False

        param_grid['eta'] = [0.01]  #[54]    train-mape:-0.450673+0.00167039    test-mape:-0.45734+0.00530681
        param_grid['max_depth'] = [13] #[78]    train-mape:-0.406378+0.00244131    test-mape:-0.456578+0.0100904
        return param_grid
    
    def adjust_cv_param(self):
        num_boost_round = 100
        early_stopping_rounds = 3
        
        kwargs = {'num_boost_round':num_boost_round, 
                  'callbacks':[xgb.callback.print_evaluation(show_stdv=True),xgb.callback.early_stop(early_stopping_rounds)]}
        return kwargs
    


if __name__ == "__main__":   
    obj= DidiXGBoostModel()
    obj.run()