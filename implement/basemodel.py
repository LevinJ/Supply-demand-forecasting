from preprocess.preparedata import PrepareData

from time import time
from evaluation.sklearnmape import mean_absolute_percentage_error
from utility.dumpload import DumpLoad

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
    def afterRun(self):
        pass
    def predictTestSet(self, dataDir):
        X_y_Df= self.getFeaturesforTestSet(dataDir)
        y_pred = self.clf.predict(X_y_Df[self.usedFeatures])
        X_y_Df['y_pred'] = y_pred
        df = X_y_Df[['start_district_id', 'time_slotid', 'y_pred']]
        filename = 'logs/'+ self.application_start_time + '_gap_prediction_result.csv'
        df.to_csv(filename , header=None, index=None)
        dumpload = DumpLoad('logs/' + self.application_start_time + '_bestestimator.pickle')
        dumpload.dump(self.clf)
        return
    def run(self):
        self.getTrainTestSet()
        self.train()
        self.test()
        self.afterRun()
        return
    
if __name__ == "__main__":   
    obj= BaseModel()
    obj.run()