from datetime import datetime
from datetime import timedelta
import pandas as pd
from utility.datafilepath import g_singletonDataFilePath
from exploredata.timeslot import singletonTimeslot
import os.path

class prepareGapCsvForPrediction:
    def __init__(self):
        return
    
    def run(self):
        self.generatePredictionCsv(g_singletonDataFilePath.getTestset1Readme())
        return
    def generatePredictionCsv(self, readmeFile):
        df = pd.read_csv(readmeFile)
        timeslotids = df.iloc[:,0].values
        res = []
        for district in range(66):
            for t in timeslotids:
                res.append([district + 1, t, singletonTimeslot.getTimeId(t)])
        df = pd.DataFrame(res, columns=['start_district_id', 'time_slotid', 'time_id'])
        df.to_csv("../data_raw/" +os.path.dirname(readmeFile).split('/')[-1] + '_gap_prediction.csv')
        return
    def load_prediction_csv(self, data_dir):
        file_path = "../data_raw/" +data_dir.split('/')[-2] + '_gap_prediction.csv'
        return pd.read_csv(file_path, index_col=0)
if __name__ == "__main__":   
    obj= prepareGapCsvForPrediction()
    obj.run()