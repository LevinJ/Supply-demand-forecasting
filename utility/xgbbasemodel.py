import xgboost as xgb
from sklearn.grid_search import ParameterGrid
from sklearn.grid_search import ParameterSampler
import pandas as pd
import matplotlib.pyplot as plt
import logging



class XGBoostBase(object):
    def __init__(self):
        self.do_cross_val = True
        self.best_score_colname_in_cv = 'test-mape-mean'
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
        model = xgb.cv(self.xgb_params, dtrain_cv, folds=cv_folds, **self.xgb_learning_params)
        best_scroe = model[self.best_score_colname_in_cv].max()
        return best_scroe
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
        
        return

    def get_paramgrid_1(self):
        pass
    def get_paramgrid_2(self, param_grid):
        return param_grid
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

    def run_grid_search(self):
        """
        This method is called by derived class to start grid search process
        """
        features,labels,cv_folds = self.getFeaturesLabel()
        dtrain_cv  = xgb.DMatrix(features, label= labels,feature_names=features.columns)
           
        parameter_iterable = self.__get_param_iterable(self.__get_param_grid())  
        kwargs = self.get_learning_params()
        for param in parameter_iterable:
            logging.info("used parameters: {}".format(param))
            bst = xgb.cv(param, dtrain_cv, folds=cv_folds,**kwargs)
            self.__add_to_resultset(param, bst)

        self.__disp_result() 
        return

    def __add_to_resultset(self, param, bst):
        max_id = bst[self.best_score_colname_in_cv].idxmax()
        self.__grid_search_result.append((param, bst.iloc[max_id][self.best_score_colname_in_cv], bst.iloc[max_id].tolist()))
        logging.info("CV score:  {}".format(bst.iloc[max_id]))
        return    
    def __disp_result(self):
        if not self.grid_search_display_result:
            return
        df = pd.DataFrame(self.__grid_search_result, columns= ['param', 'result', 'otherinfo'])
        logging.info( '\nall para search results:')
        logging.info("{}".format( df.values))
        best_score_id = df['result'].idxmax()
        logging.info( '\nbest parameters:')
#         logging.info("{}".format(df.iloc[best_score_id]['param']))
#         logging.info("{}".format( df.iloc[best_score_id]['result']))
        logging.info("{}".format( df.iloc[best_score_id].values))
        df.to_csv('temp/__grid_search_result.csv')
        
        return