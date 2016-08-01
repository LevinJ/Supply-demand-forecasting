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
from utility.modelframework import ModelFramework
from utility.xgbbasemodel import XGBoostGridSearch
from evaluation.sklearnmape import mean_absolute_percentage_error_xgboost_cv
from utility.xgbbasemodel import XGBoostBase




class DidiXGBoostModel(XGBoostBase, PrepareData, XGBoostGridSearch):
    def __init__(self):
        PrepareData.__init__(self)
        XGBoostGridSearch.__init__(self)
        XGBoostBase.__init__(self)
        

        self.do_cross_val = False
#         self.holdout_split = HoldoutSplitMethod.IMITTATE_TEST2_MIN #[78]    train-mape:-0.373462+0.00492894    test-mape:-0.46214+0.0114662
#         self.holdout_split = HoldoutSplitMethod.IMITTATE_TEST2_PLUS1 # [79]    train-mape:-0.396072+0.00363566    test-mape:-0.459982+0.0100845
#         self.holdout_split = HoldoutSplitMethod.IMITTATE_TEST2_PLUS3 #[78]    train-mape:-0.411597+0.00219856    test-mape:-0.454906+0.0124385
#         self.holdout_split = HoldoutSplitMethod.IMITTATE_TEST2_PLUS4#[78]    train-mape:-0.411597+0.00219856    test-mape:-0.454906+0.0124385
#         self.holdout_split = HoldoutSplitMethod.IMITTATE_TEST2_PLUS6# [82]    train-mape:-0.421504+0.00191357    test-mape:-0.453868+0.0116971
#         self.holdout_split = HoldoutSplitMethod.IMITTATE_TEST2_FULL#[81]    train-mape:-0.428109+0.000703088    test-mape:-0.451429+0.0114696
        return

    def run_croos_validation(self):
         
        # Use default parameters only
        param = {'max_depth':14, 'eta':0.02, 'silent':1, 'objective':'reg:linear' }
        
        #Run 100 rounds, just to ensure it's sufficiently trained
        num_boost_round = 100
        early_stopping_rounds = 3
         
        # specify validations set to watch performance
        xgb.cv(param, self.dtrain_cv, num_boost_round=num_boost_round,
               callbacks=[xgb.callback.print_evaluation(show_stdv=True),xgb.callback.early_stop(early_stopping_rounds)], **self.folds_params)

        return 
    def run_train_validation(self):

        # specify parameters via map, definition are same as c++ version
        param = {'max_depth':14, 'eta':0.02, 'silent':1, 'objective':'reg:linear' }
#         param = { 'silent':1}
         
        # specify validations set to watch performance
        evals  = [(self.dtrain,'train'),(self.dvalidation,'eval')]
        num_boost_round = 100
        early_stopping_rounds=3
        
        xgb.train(param, self.dtrain, num_boost_round=num_boost_round, evals = evals, early_stopping_rounds=early_stopping_rounds,**self.folds_params)
         
        print "features used:\n {}".format(self.usedFeatures)
         
        return
    def get_model_input(self):
        
        if self.do_cross_val is None or self.do_cross_val: 
            # for cross validation
            features,labels,self.cv_folds = self.getFeaturesLabel()
            self.dtrain_cv  = xgb.DMatrix(features, label= labels,feature_names=features.columns)
            #for grid search
            self.folds_params = {'folds':self.cv_folds, 'feval':mean_absolute_percentage_error_xgboost_cv}
            return
        
        # for train validation
        x_train, y_train,x_validation,y_validation = self.get_train_validationset()
        self.dtrain = xgb.DMatrix(x_train, label= y_train,feature_names=x_train.columns)
        self.dvalidation = xgb.DMatrix(x_validation, label= y_validation,feature_names=x_validation.columns)
        
        self.folds_params = {'feval':mean_absolute_percentage_error_xgboost}
        
        return
    def adjust_intial_param(self):
        """
        This method must be overriden by derived class when its objective is not reg:linear
        """
        param = {'max_depth':6, 'eta':0.1,  'min_child_weight':1,'silent':1, 
                 'objective':'reg:linear','colsample_bytree':0.8,'subsample':0.8,'min_child_weight':1 }
        return param
    
    def adjust_param(self, param_grid):
        """
        This method must be overriden by derived class if it intends to fine tune parameters
        """
        self.ramdonized_search_enable = False
        self.randomized_search_n_iter = 2
        self.grid_search_display_result = True

        param_grid['eta'] = [0.01, 0.05]  #[54]    train-mape:-0.450673+0.00167039    test-mape:-0.45734+0.00530681
        param_grid['max_depth'] = [13] #[78]    train-mape:-0.406378+0.00244131    test-mape:-0.456578+0.0100904
        return param_grid
    
    def adjust_cv_param(self):
        """e
        This method must be overriden by derived class if it intends to fine tune parameters
        """
        num_boost_round = 100
        early_stopping_rounds = 3
        
        kwargs = {'num_boost_round':num_boost_round, 
                  'callbacks':[xgb.callback.print_evaluation(show_stdv=True),xgb.callback.early_stop(early_stopping_rounds)]}
        return kwargs
    


if __name__ == "__main__":   
    obj= DidiXGBoostModel()
    obj.run()