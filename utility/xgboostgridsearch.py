from sklearn.grid_search import ParameterGrid
from sklearn.grid_search import ParameterSampler
import xgboost as xgb
import pandas as pd



    
class BaseSearchCV(object):
    def __init__(self, param_grid, kwargs):
        self.param_grid = param_grid
        kwargs['callbacks'].append(self.get_early_stop_metric())
        self.kwargs = kwargs
        self.search_result = []
        self.early_stop_metric = None
        self.display_result = True
        
        return
    def get_early_stop_metric(self):
    
        def callback(env):
            if self.early_stop_metric is None:
                self.early_stop_metric = env.evaluation_result_list[-1][0]
                self.early_stop_metric = self.early_stop_metric + '-mean'
            return
        return callback
    def cv(self, dtrain=None):
        return self._cv(dtrain, ParameterGrid(self.param_grid))
    def _cv(self, dtrain, parameter_iterable):

        for param in parameter_iterable:
            print param
            bst = xgb.cv(param, dtrain, **self.kwargs)
            self.add_to_resultset(param, bst)
        self.disp_result()
    def add_to_resultset(self, param, bst):
        max_id = bst[self.early_stop_metric].idxmax()
        self.search_result.append((param, bst.iloc[max_id][self.early_stop_metric], bst.iloc[max_id].tolist()))
        return    
    def disp_result(self):
        if not self.display_result:
            return
        df = pd.DataFrame(self.search_result, columns= ['param', 'result', 'otherinfo'])
        print '\nall para search results:'
        print df
        best_score_id = df['result'].idxmax()
        print '\nbest parameters:'
        print df.iloc[best_score_id]['param']
        print df.iloc[best_score_id]['result']
        print df.iloc[best_score_id]['otherinfo']
        df.to_csv('temp/search_result.csv')
        return
    def run(self):

        return

    
class GridSearchCV(BaseSearchCV):
    def __init__(self, param_grid, kwargs):
        BaseSearchCV.__init__(self, param_grid, kwargs)
        return

class RandomizedSearchCV(BaseSearchCV):
    def __init__(self, param_grid, n_iter=10,random_state = None, kwargs=None):
        self.n_iter = n_iter
        self.random_state = random_state
        BaseSearchCV.__init__(self, param_grid, kwargs)
        return
    def cv(self, dtrain=None):
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
        return self._cv(dtrain, sampled_params)
    
if __name__ == "__main__":   
    obj= GridSearchCV()
    obj.run()