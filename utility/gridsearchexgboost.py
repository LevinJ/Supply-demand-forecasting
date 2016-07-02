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
        self.grid_search_display_result = False

        param_grid['eta'] = [0.01]  #[54]    train-mape:-0.450673+0.00167039    test-mape:-0.45734+0.00530681
        param_grid['max_depth'] = [13] #[78]    train-mape:-0.406378+0.00244131    test-mape:-0.456578+0.0100904
        return param_grid
    def adjust_cv_param(self):
        """
        This method must be overriden by derived class if it intends to fine tune parameters
        """
        num_boost_round = 10
        early_stopping_rounds = 3
        
        kwargs = {'num_boost_round':num_boost_round, 
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
        kwargs['callbacks'].append(self.__get_early_stop_metric())
        return kwargs
    def run_grid_search(self):
        """
        This method is called by derived class to start grid search process
        """
           
        parameter_iterable = self.__get_param_iterable(self.__get_param_grid())  
        kwargs = self.__get_kwargs(self.folds_params)
        for param in parameter_iterable:
            print param
            bst = xgb.cv(param, self.dtrain, **kwargs)
            self.__add_to_resultset(param, bst)
        self.__disp_result() 
        return
    def __get_early_stop_metric(self):
    
        def callback(env):
            if self.__grid_search_early_stop_metric is None:
                self.__grid_search_early_stop_metric = env.evaluation_result_list[-1][0]
                self.__grid_search_early_stop_metric = self.__grid_search_early_stop_metric + '-mean'
            return
        return callback
    def __add_to_resultset(self, param, bst):
        max_id = bst[self.__grid_search_early_stop_metric].idxmax()
        self.__grid_search_result.append((param, bst.iloc[max_id][self.__grid_search_early_stop_metric], bst.iloc[max_id].tolist()))
        return    
    def __disp_result(self):
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