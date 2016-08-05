import xgboost as xgb
from sklearn.grid_search import ParameterGrid
from sklearn.grid_search import ParameterSampler
import pandas as pd
import matplotlib.pyplot as plt



class XGBoostBase(object):
    def __init__(self):
        self.do_cross_val = True
        return
    def run(self):
#         self.get_model_input()
        if self.do_cross_val is None:
            return self.run_grid_search()
        if self.do_cross_val:
            return self.run_croos_validation()
        return self.run_train_validation()
    def run_croos_validation(self):
        
        features,labels,cv_folds = self.getFeaturesLabel()
        dtrain_cv  = xgb.DMatrix(features, label= labels,feature_names=features.columns)
        self.set_xgb_parameters()

        # specify validations set to watch performance
        xgb.cv(self.xgb_params, dtrain_cv, folds=cv_folds, **self.xgb_learning_params)
        return 
    def set_xgb_parameters(self):
        self.xgb_params = {'silent':1}
        self.xgb_learning_params = {}
        return
    def run_train_validation(self):
        x_train, y_train,x_validation,y_validation = self.get_train_validationset()
        dtrain = xgb.DMatrix(x_train, label= y_train,feature_names=x_train.columns)
        dvalidation = xgb.DMatrix(x_validation, label= y_validation,feature_names=x_validation.columns)
        self.set_xgb_parameters()
        
        evals=[(dtrain,'train'),(dvalidation,'eval')]
        model = xgb.train(self.xgb_params, dtrain, evals=evals, **self.xgb_learning_params)
        xgb.plot_importance(model)
        plt.show()
         
        print "features used:\n {}".format(self.get_used_features())
         
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

    def get_paramgrid_1(self):
        pass
    def get_paramgrid_2(self, param_grid):
        pass
    def get_learning_params(self):
        pass
    def __get_param_iterable(self, param_grid):
        if self.ramdonized_search_enable:
            parameter_iterable = ParameterSampler(param_grid,
                                          self.randomized_search_n_iter,
                                          random_state=self.ramdonized_search_random_state)
        else:
            parameter_iterable = ParameterGrid(param_grid)
                 
        return parameter_iterable
    def __get_param_grid(self):
        param_grid = self.get_paramgrid_1()
        param_grid = self.get_paramgrid_2(param_grid) 
        return param_grid
    def __get_kwargs(self):
        # specify validations set to watch performance
        kwargs = self.get_learning_params()
        kwargs['callbacks'].append(self.__get_early_stop_metric())
        return kwargs
    def run_grid_search(self):
        """
        This method is called by derived class to start grid search process
        """
        features,labels,cv_folds = self.getFeaturesLabel()
        dtrain_cv  = xgb.DMatrix(features, label= labels,feature_names=features.columns)
           
        parameter_iterable = self.__get_param_iterable(self.__get_param_grid())  
        kwargs = self.__get_kwargs()
        for param in parameter_iterable:
            print param
            bst = xgb.cv(param, dtrain_cv, folds=cv_folds,**kwargs)
            self.__add_to_resultset(param, bst)
#             xgb.callback.early_stop.cleanup()# clear the callb ack state
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