from  evaluation.evaluate import Evaluate
import pandas as pd
from utility.datafilepath import g_singletonDataFilePath

class Statistics(Evaluate):
    def __init__(self):
        return
    def getPrevSlots(self, timeslot):
        res = []
        date = timeslot.split('-')[:3]
        slot = int(timeslot.split('-')[-1])
        preNum = 3
        for i in range(preNum):
            item = date + [str(slot -i -1)]
            res.append("-".join(item))
        return res
    def generatePrediction(self, allOderFilePath, testSlots,resultFileName):
        res = []
        df = pd.read_csv(allOderFilePath)
        testDistricts = self.generateTestDistrict()
        for timeslot in testSlots:
            for district in testDistricts:
                prevSlots = self.getPrevSlots(timeslot)
                sel = df['time_slotid'].isin(prevSlots) & (df['start_district_id'] == district)
                item = df[sel]['gap']    
                assert item.shape[0] <= 3, item.shape[0]
                if item.empty:
                    # no previous records found, assume zero
                    item = 0
                else:
                    item = item.mean()
                res.append((district, timeslot, item))
        df = pd.DataFrame(res, columns=['start_district_id', 'time_slotid', 'gap'])
        self.saveResultCsv(df, resultFileName)
        return
        
    def generatePrediction_0(self):
        testSlots = self.generateSlotSet_0()
        self.generatePrediction(g_singletonDataFilePath.getGapCsv_Train(), testSlots, 'prediction_0.csv')
        return
    def generatePrediction_1(self):
        testSlots = self.generateSlotSet_1()
        self.generatePrediction(g_singletonDataFilePath.getGapCsv_Test1(), testSlots, 'temp/prediction_1.csv')
        return
    def run(self):
        assert  ['2016-01-22-45','2016-01-22-44','2016-01-22-43'] == self.getPrevSlots('2016-01-22-46')
        self.generatePrediction_1()
#         self.generatePrediction_0()
#         self.calFinalResult(0)
        return
if __name__ == "__main__":   
    obj= Statistics()
    obj.run()