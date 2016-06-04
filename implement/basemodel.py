from preprocess.preparedata import PrepareData

from time import time
from evaluation.sklearnmape import mean_absolute_percentage_error
from sklearn.metrics import mean_squared_error

class BaseModel(PrepareData):
    def __init__(self):
        PrepareData.__init__(self)
        self.setClf()
        
        return
    def setClf(self):
        pass
    def afterTrain(self):
        pass
    def train(self):
        print "Training {}...".format(self.clf.__class__.__name__)
        t0 = time()
        self.clf.fit(self.X_train, self.y_train)
        print "train:", round(time()-t0, 3), "s"
        self.afterTrain()
        return
    def dispFeatureImportance(self):
        pass
    def test(self):
        t0 = time()
        y_pred_train = self.clf.predict(self.X_train)
        y_pred_test = self.clf.predict(self.X_test)
        print "features used:\n {}".format(self.usedFeatures)
        print "MAPE for training set: {}".format(mean_absolute_percentage_error(self.y_train, y_pred_train))
        print "MAPE for testing set: {}".format(mean_absolute_percentage_error(self.y_test, y_pred_test))
#         print "MSE for training set: {}".format(mean_squared_error(self.y_train, y_pred_train))
#         print "MSE for testing set: {}".format(mean_squared_error(self.y_test, y_pred_test))
        print "test:", round(time()-t0, 3), "s"
        return
    def run(self):
        self.getTrainTestSet()
        self.train()
        self.test()
        return
    
if __name__ == "__main__":   
    obj= BaseModel()
    obj.run()