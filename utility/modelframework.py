from utility.runtype import RunType


class ModelFramework(object):
    """
    This is meant to be a base class that should be inherited by concrete models,and they should implement
    get_model_input
    run_train_validation
    run_croos_validation
    run_grid_search
    """
    def __init__(self):
        self.dtrain = None
        self.folds_params = None 
        self.run_type = RunType.RUN_GRID_SEARCH
        return
    def get_model_input(self):
        """
        All these parameter should be configured by derived class
        self.dtrain, the data to be trained
        self.folds_params, cross validation folds index, and evaluation metric
        self.run_type, how to run the model
        """
        self.dtrain = None
        self.folds_params = None 
        self.run_type = None
        self.folds_id_used = None
        return
    def run(self):
        run_dict = {}
        run_dict[RunType.RUN_TRAIN_VALIDATION] = self.run_train_validation
        run_dict[RunType.RUN_GRID_SEARCH] = self.run_grid_search
        run_dict[RunType.RUN_CROSS_VALIDATION] = self.run_croos_validation
       
        self.get_model_input()
        run_dict[self.run_type]()        
         
        return
    def run_croos_validation(self):
        pass
    def run_train_validation(self):
        pass
    def run_grid_search(self):
        pass