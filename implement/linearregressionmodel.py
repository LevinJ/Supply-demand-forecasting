from basemodel import BaseModel
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression

class LinearRegressionModel(BaseModel):
    def __init__(self):
        BaseModel.__init__(self)
        self.usedFeatures = ['gap1', 'gap2', 'gap3']
        return
    def setClf(self):
#         self.clf = Ridge(alpha=0.0000001, tol=0.0000001)
        self.clf = LinearRegression()
        return
    def afterTrain(self):
        print "self.clf.coef_:\n{}".format(self.clf.coef_)
        print "self.clf.intercept_:\n{}".format(self.clf.intercept_)
        return




if __name__ == "__main__":   
    obj= LinearRegressionModel()
    obj.run()