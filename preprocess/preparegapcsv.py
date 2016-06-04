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
        df.to_csv(os.path.dirname(readmeFile) + "/" + g_singletonDataFilePath.getGapPredictionFileName())
        return
if __name__ == "__main__":   
    obj= prepareGapCsvForPrediction()
    obj.run()