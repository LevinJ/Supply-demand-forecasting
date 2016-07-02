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
# from utility.xgboostgridsearch import GridSearchCV
# from utility.xgboostgridsearch import RandomizedSearchCV
from utility.finetunexgboost import Fine_Tune_XGBoost



    
class XGBoostModel(Fine_Tune_XGBoost):
    def __init__(self):
        Fine_Tune_XGBoost.__init__(self)
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
    def adjust_cv_param(self):
        self.use_randomized_search = False
        self.n_iter_randomized_search = 3
        
        
        num_boost_round = 3
        early_stopping_rounds = 3
        folds = 5
        
        kwargs = {'num_boost_round':num_boost_round, 'folds':folds,
                  'callbacks':[xgb.callback.print_evaluation(show_stdv=True),xgb.callback.early_stop(early_stopping_rounds)]}
        return kwargs
#     def run_grid_search(self, dtrain,cv):
#         self.tune(dtrain, cv)
# #         param_grid = {'max_depth':[7,8], 'eta':[0.01], 'silent':[1], 'objective':['reg:linear'] }
#         param_grid = {'max_depth':range(5,15), 'eta':[0.01, 0.02,0.005], 'silent':[1], 'objective':['reg:linear'] }
#         num_boost_round = 500
#         early_stopping_rounds = 3
#         do_random_gridsearch = True 
#         n_iter=10
#         
#         if do_random_gridsearch:
#             grid = RandomizedSearchCV( param_grid,  cv=cv, num_boost_round = num_boost_round, 
#                                    early_stopping_rounds=early_stopping_rounds,feval = mean_absolute_percentage_error_xgboost, n_iter=n_iter)
#         else:
#             grid = GridSearchCV( param_grid,  cv=cv, num_boost_round = num_boost_round, 
#                                    early_stopping_rounds=early_stopping_rounds,feval = mean_absolute_percentage_error_xgboost)
#         grid.fit(dtrain)
#         return

class DidiXGBoostModel( BaseModel, XGBoostModel):
    def __init__(self):
        BaseModel.__init__(self)
        XGBoostModel.__init__(self)
        self.holdout_split = HoldoutSplitMethod.IMITTATE_TEST2_MIN #[78]    train-mape:-0.373462+0.00492894    test-mape:-0.46214+0.0114662
        self.holdout_split = HoldoutSplitMethod.IMITTATE_TEST2_PLUS1 # [79]    train-mape:-0.396072+0.00363566    test-mape:-0.459982+0.0100845
        self.holdout_split = HoldoutSplitMethod.IMITTATE_TEST2_PLUS3 #[78]    train-mape:-0.411597+0.00219856    test-mape:-0.454906+0.0124385
        self.holdout_split = HoldoutSplitMethod.IMITTATE_TEST2_PLUS4#[78]    train-mape:-0.411597+0.00219856    test-mape:-0.454906+0.0124385
        self.holdout_split = HoldoutSplitMethod.IMITTATE_TEST2_PLUS6# [82]    train-mape:-0.421504+0.00191357    test-mape:-0.453868+0.0116971
        self.holdout_split = HoldoutSplitMethod.IMITTATE_TEST2_FULL#[81]    train-mape:-0.428109+0.000703088    test-mape:-0.451429+0.0114696
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
        
        self.run_grid_search(dtrain,{'folds':cv, 'feval':mean_absolute_percentage_error_xgboost})     
        
         
        return
    def adjust_intial_param(self):
        param = {'max_depth':6, 'eta':0.1,  'min_child_weight':1,'silent':1, 
                 'objective':'reg:linear','colsample_bytree':0.8,'subsample':0.8,'min_child_weight':1 }
        return param
    
    def adjust_param(self, param_grid):
        self.use_randomized_search = False
        self.n_iter_randomized_search = 3
        self.display_result = False

        param_grid['eta'] = [0.01]  #[54]    train-mape:-0.450673+0.00167039    test-mape:-0.45734+0.00530681
        param_grid['max_depth'] = [13] #[78]    train-mape:-0.406378+0.00244131    test-mape:-0.456578+0.0100904
        return param_grid
    
    def adjust_cv_param(self):
        num_boost_round = 200
        early_stopping_rounds = 10
        
        kwargs = {'num_boost_round':num_boost_round, 
                  'callbacks':[xgb.callback.print_evaluation(show_stdv=True),xgb.callback.early_stop(early_stopping_rounds)]}
        return kwargs
    


if __name__ == "__main__":   
    obj= DidiXGBoostModel()
    obj.run()