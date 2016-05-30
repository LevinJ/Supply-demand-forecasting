from  evaluation.evaluate import Evaluate
import pandas as pd
import math

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
    def generatePrediction_0(self):
        res = []
        allOderFilePath = '../data/citydata/season_1/training_data/order_data/temp/allorders.csv'
        df = pd.read_csv(allOderFilePath)
        testSlots = self.generateSlotSet_0()
        testDistricts = self.generateTestDistrict()
        for timeslot in testSlots:
            for district in testDistricts:
                prevSlots = self.getPrevSlots(timeslot)
                sel = df['time_slotid'].isin(prevSlots) & (df['start_district_id'] == district)
                item = df[sel]['missed_request']    
                assert item.shape[0] <= 3, item.shape[0]
                if item.empty:
                    # no previous records found, assume zero
                    item = 0
                else:
                    item = item.mean()
                res.append((district, timeslot, item))
        df = pd.DataFrame(res, columns=['start_district_id', 'time_slotid', 'missed_request'])
        df['timeslotrank'] = df['time_slotid'].map(lambda x: "-".join(x.split('-')[:3] + [x.split('-')[-1].zfill(3)]))
        df = df.sort_values(by = ['timeslotrank','start_district_id'])
        df.to_csv('prediction_0.csv', columns=['start_district_id', 'time_slotid', 'missed_request'], header=None, index=None)
        return
    def run(self):
        assert  ['2016-01-22-45','2016-01-22-44','2016-01-22-43'] == self.getPrevSlots('2016-01-22-46')
#         self.generatePrediction_0()
        self.calFinalResult(0)
        return
if __name__ == "__main__":   
    obj= Statistics()
    obj.run()