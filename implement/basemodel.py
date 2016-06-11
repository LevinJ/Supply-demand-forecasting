from preprocess.preparedata import PrepareData

from time import time
from evaluation.sklearnmape import mean_absolute_percentage_error
from utility.dumpload import DumpLoad
import numpy as np
from datetime import datetime
from utility.datafilepath import g_singletonDataFilePath

class BaseModel(PrepareData):
    def __init__(self):
        PrepareData.__init__(self)
        self.setClf()
        self.application_start_time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        self.save_final_model =False
        
        return
    def setClf(self):
        pass
    def afterTrain(self):
        self.dispFeatureImportance()
        return
    def train(self):
        print "Training {}...".format(self.clf.__class__.__name__)
        t0 = time()
        self.clf.fit(self.X_train, self.y_train)
        print "train:", round(time()-t0, 3), "s"
        self.afterTrain()
        return
    def save_model(self):
        if not self.save_final_model:
            return
        dumpload = DumpLoad('logs/' + self.application_start_time + '_estimator.pickle')
        dumpload.dump(self)
        self.predictTestSet(g_singletonDataFilePath.getTest1Dir())
        return
    def dispFeatureImportance(self):
        if not hasattr(self.clf, 'feature_importances_'):
            return
        features_list = np.asanyarray(self.usedFeatures)
        sortIndexes = self.clf.feature_importances_.argsort()[::-1]
        features_rank = features_list[sortIndexes]
        num_rank = self.clf.feature_importances_[sortIndexes]
        print "Ranked features: {}".format(features_rank)
        print "Ranked importance: {}".format(num_rank)
        return
    def test(self):
        t0 = time()
        y_pred_train = self.clf.predict(self.X_train)
        y_pred_test = self.clf.predict(self.X_test)
        print "features used:\n {}".format(self.usedFeatures)
        print "MAPE for training set: {}".format(mean_absolute_percentage_error(self.y_train, y_pred_train, dateslot_num = self.dateslot_train_num))
        print "MAPE for testing set: {}".format(mean_absolute_percentage_error(self.y_test, y_pred_test, dateslot_num = self.dateslot_test_num))
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
        
        return
    def run(self):
        self.getTrainTestSet()
        self.train()
        self.test()
        self.afterRun()
        self.save_model()
        return
    
if __name__ == "__main__":   
    obj= BaseModel()
    obj.run()