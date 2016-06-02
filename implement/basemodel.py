from preprocess.preparedata import PrepareData

from time import time
from evaluation.sklearnmape import mean_absolute_percentage_error

class BaseModel(PrepareData):
    def __init__(self):
        PrepareData.__init__(self)
        
        return
    def setClf(self):
        pass
    def train(self):
        self.setClf()
        print "Training {}...".format(self.clf.__class__.__name__)
        t0 = time()
        self.clf.fit(self.X_train, self.y_train)
        print "train:", round(time()-t0, 3), "s"
        return
    def test(self):
        t0 = time()
        print "MAPE for training set: {}".format(mean_absolute_percentage_error(self.y_train, self.clf.predict(self.X_train)))
        print "MAPE for testing set: {}".format(mean_absolute_percentage_error(self.y_test, self.clf.predict(self.X_test)))
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