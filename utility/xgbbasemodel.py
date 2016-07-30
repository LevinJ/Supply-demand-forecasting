import xgboost as xgb
from sklearn.grid_search import ParameterGrid
from sklearn.grid_search import ParameterSampler
import pandas as pd



class XGBoostBase(object):
    def __init__(self):
        self.do_cross_val = True
        return
    def run(self):
        self.get_model_input()
        if self.do_cross_val is None:
            return self.run_grid_search()
        if self.do_cross_val:
            return self.run_croos_validation()
        return self.run_train_validation()
    def run_croos_validation(self):
         
        # Use default parameters only
        param = {'silent':1}
        
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
            self.folds_params = None
            return
        
        # for train validation
        x_train, y_train,x_validation,y_validation = self.get_train_validationset()
        self.dtrain = xgb.DMatrix(x_train, label= y_train,feature_names=x_train.columns)
        self.dvalidation = xgb.DMatrix(x_validation, label= y_validation,feature_names=x_validation.columns)
        
        self.folds_params = None
        
       
        return
    
class XGBoostGridSearch(object):
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
            bst = xgb.cv(param, self.dtrain_cv, **kwargs)
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