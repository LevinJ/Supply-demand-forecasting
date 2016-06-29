from sklearn.grid_search import ParameterGrid
from sklearn.grid_search import ParameterSampler
import xgboost as xgb
import pandas as pd


    
class BaseSearchCV(object):
    def __init__(self, param_grid, cv=None, num_boost_round = 10, early_stopping_rounds=3, feval = None):
        self.param_grid = param_grid
        self.cv = cv
        self.num_boost_round = num_boost_round
        self.early_stopping_rounds = early_stopping_rounds
        self.feval = feval
        self.search_result = []
        
        return
    def fit(self, dtrain=None):
        """Run fit with all sets of parameters.

        Parameters
        ----------

        X : array-like, shape = [n_samples, n_features]
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.

        y : array-like, shape = [n_samples] or [n_samples, n_output], optional
            Target relative to X for classification or regression;
            None for unsupervised learning.

        """
        return self._fit(dtrain, ParameterGrid(self.param_grid))
    def _fit(self, dtrain, parameter_iterable):
        """Actual fitting,  performing the search over parameters."""

        for param in parameter_iterable:
            print param
            bst = xgb.cv(param, dtrain, num_boost_round=self.num_boost_round,  feval = self.feval, folds = self.cv,callbacks=[xgb.callback.print_evaluation(show_stdv=True),
                        xgb.callback.early_stop(self.early_stopping_rounds)])
            self.add_to_resultset(param, bst)
        self.disp_result()
    def add_to_resultset(self, param, bst):
        max_id = bst['test-mape-mean'].idxmax()
        self.search_result.append((param, bst.iloc[max_id]['test-mape-mean'], bst.iloc[max_id].tolist()))
        return    
    def disp_result(self):
        df = pd.DataFrame(self.search_result, columns= ['param', 'result', 'otherinfo'])
        print 'final result:\n', df
        print 'best parameters:\n', df.iloc[df['result'].idxmax()]
        df.to_csv('temp/search_result.csv')
        return
    def run(self):

        return

    
class GridSearchCV(BaseSearchCV):
    def __init__(self, param_grid, cv=None, num_boost_round = 10, early_stopping_rounds=3, feval = None):
        BaseSearchCV.__init__(self, param_grid, cv, num_boost_round, early_stopping_rounds, feval)
        return

class RandomizedSearchCV(BaseSearchCV):
    def __init__(self, param_grid, cv=None, num_boost_round = 10, early_stopping_rounds=3, feval = None,n_iter=10,random_state=None):
        self.n_iter = n_iter
        self.random_state = random_state
        BaseSearchCV.__init__(self, param_grid, cv, num_boost_round, early_stopping_rounds, feval)
        return
    def fit(self, dtrain=None):
        """Run fit on the estimator with randomly drawn parameters.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Training vector, where n_samples in the number of samples and
            n_features is the number of features.

        y : array-like, shape = [n_samples] or [n_samples, n_output], optional
            Target relative to X for classification or regression;
            None for unsupervised learning.

        """
        sampled_params = ParameterSampler(self.param_grid,
                                          self.n_iter,
                                          random_state=self.random_state)
        return self._fit(dtrain, sampled_params)
    
if __name__ == "__main__":   
    obj= GridSearchCV()
    obj.run()