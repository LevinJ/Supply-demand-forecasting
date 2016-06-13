import sys
import os
sys.path.insert(0, os.path.abspath('..')) 
# from pprint import pprint as p
# p(sys.path)

# print os.environ['PYTHONPATH'].split(os.pathsep)
from basemodel import BaseModel
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
from sklearn.pipeline import Pipeline




class LinearRegressionModel(BaseModel):
    def __init__(self):
        BaseModel.__init__(self)
        self.save_final_model = False
        self.do_cross_val = True
        return
    def get_train_validation_foldid(self):
        return -1
    def setClf(self):
#         self.clf = Ridge(alpha=0.0000001, tol=0.0000001)
        clf = LinearRegression()
        min_max_scaler = preprocessing.MinMaxScaler()
        self.clf = Pipeline([('scaler', min_max_scaler), ('estimator', clf)])
        return
    def afterTrain(self):
        print "self.clf.named_steps['estimator'].coef_:\n{}".format(self.clf.named_steps['estimator'].coef_)
        print "self.clf.named_steps['estimator'].intercept_:\n{}".format(self.clf.named_steps['estimator'].intercept_)
        return




if __name__ == "__main__":   
    obj= LinearRegressionModel()
    obj.run()