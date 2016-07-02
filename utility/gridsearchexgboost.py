import xgboost as xgb
from sklearn.grid_search import ParameterGrid
from sklearn.grid_search import ParameterSampler
import pandas as pd




class GridSearchXGBoost(object):
    def __init__(self):
        self.ramdonized_search_enable = False
        self.randomized_search_n_iter = 10
        self.ramdonized_search_random_state = None
        
        self.grid_search_display_result = True
        self.__grid_search_result = []
        self.__grid_search_early_stop_metric = None
        
        return
    def run(self):
        self.tune()
        return
    def adjust_intial_param(self):
        param = {'max_depth':6, 'eta':0.1,  'min_child_weight':1,'silent':1, 
                 'objective':'binary:logistic','colsample_bytree':0.8,'subsample':0.8,'min_child_weight':1,'eval_metric':'auc' }
        return param
    def adjust_param(self, param):
        pass
    def adjust_cv_param(self):
        
        num_boost_round = 100
        early_stopping_rounds = 3
        nfold = 5
        
        kwargs = {'num_boost_round':num_boost_round, 'nfold':nfold,
                  'callbacks':[xgb.callback.print_evaluation(show_stdv=True),xgb.callback.early_stop(early_stopping_rounds)]}
        return kwargs
    def __get_param_iterable(self, param_grid):
        if self.ramdonized_search_enable:
            parameter_iterable = ParameterSampler(param_grid,
                                          self.randomized_search_n_iter,
                                          random_state=self.ramdonized_search_random_state)
        else:
            parameter_iterable = ParameterGrid(param_grid)
                 
        return parameter_iterable
    def __get_param_grid(self):
        param = self.adjust_intial_param()
        param_grid = {}
        for key, value in param.iteritems():
            param_grid[key] = [value]
        
        param_grid = self.adjust_param(param_grid) 
        return param_grid
    def __get_kwargs(self, folds_params):
        # specify validations set to watch performance
        kwargs = self.adjust_cv_param()
        # add the cv related parameters
        for key, value in folds_params.iteritems():
            kwargs[key] = value 
        kwargs['callbacks'].append(self.get_early_stop_metric())
        return kwargs
    def run_grid_search(self):
           
        parameter_iterable = self.__get_param_iterable(self.__get_param_grid())  
        kwargs = self.__get_kwargs(self.folds_params)
        for param in parameter_iterable:
            print param
            bst = xgb.cv(param, self.dtrain, **kwargs)
            self.add_to_resultset(param, bst)
        self.disp_result() 
        return
    def get_early_stop_metric(self):
    
        def callback(env):
            if self.__grid_search_early_stop_metric is None:
                self.__grid_search_early_stop_metric = env.evaluation_result_list[-1][0]
                self.__grid_search_early_stop_metric = self.__grid_search_early_stop_metric + '-mean'
            return
        return callback
    def add_to_resultset(self, param, bst):
        max_id = bst[self.__grid_search_early_stop_metric].idxmax()
        self.__grid_search_result.append((param, bst.iloc[max_id][self.__grid_search_early_stop_metric], bst.iloc[max_id].tolist()))
        return    
    def disp_result(self):
        if not self.grid_search_display_result:
            return
        df = pd.DataFrame(self.__grid_search_result, columns= ['param', 'result', 'otherinfo'])
        print '\nall para search results:'
        print df
        best_score_id = df['result'].idxmax()
        print '\nbest parameters:'
        print df.iloc[best_score_id]['param']
        print df.iloc[best_score_id]['result']
        print df.iloc[best_score_id]['otherinfo']
        df.to_csv('temp/__grid_search_result.csv')
        return