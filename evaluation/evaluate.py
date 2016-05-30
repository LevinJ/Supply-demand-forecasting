import pandas as pd
import numpy as np
from datetime import datetime
from datetime import timedelta
import math

class GenerateResultCsv:
    def __init__(self):
        return
    def generateTestDate_0(self):
        startDate = datetime.strptime('2016-01-01', '%Y-%m-%d')
        
        res = []
        for i in range(21):
            deltatime = timedelta(days = i)
            item = (startDate + deltatime).date()
            res.append(str(item))
        return res
    def generateTestDate_1(self):
        startDate = datetime.strptime('2016-01-22', '%Y-%m-%d')
        
        res = []
        for i in range(5):
            deltatime = timedelta(days = 2*i)
            item = (startDate + deltatime).date()
            res.append(str(item))
        return res
    def generateSlotSet(self, testDates, slots):
        res = []
        for testDate in testDates:
            for slot in slots:
                res.append(testDate + '-'+ str(slot))
        return res
    def generateSlotSet_0(self):
        testDates = self.generateTestDate_0()
        slots = [46,58,70,82,94,106,118,130,142]
        return self.generateSlotSet(testDates, slots)
    def generateSlotSet_1(self):
        testDates = self.generateTestDate_1()
        slots = [46,58,70,82,94,106,118,130,142]
        return self.generateSlotSet(testDates, slots)
    def generateTestDistrict(self):
        return [i+ 1 for i in range(66)]
    def generatePrediction_0(self):
        testSlots = self.generateSlotSet_0()
        allOderFilePath = '../data/citydata/season_1/training_data/order_data/temp/allorders.csv'
        df = pd.read_csv(allOderFilePath)
        df = df.loc[df['time_slotid'].isin(testSlots)]
        self.saveResultCsv(df, 'prediction_0.csv')
        #map 2016-01-22-1 to 2016-01-22-001
#         df['timeslotrank'] = df['time_slotid'].map(lambda x: "-".join(x.split('-')[:3] + [x.split('-')[-1].zfill(3)]))
#         df = df.sort_values(by = ['timeslotrank','start_district_id'])
#         df.to_csv('prediction_0.csv', columns=['start_district_id', 'time_slotid', 'missed_request'], header=None, index=None)
        return
    def saveResultCsv(self, df, filename):
        #map 2016-01-22-1 to 2016-01-22-001
        df['timeslotrank'] = df['time_slotid'].map(lambda x: "-".join(x.split('-')[:3] + [x.split('-')[-1].zfill(3)]))
        df = df.sort_values(by = ['timeslotrank','start_district_id'])
        df.to_csv(filename, columns=['start_district_id', 'time_slotid', 'missed_request'], header=None, index=None)
        return
    def generateActual_0(self):
        testSlots = self.generateSlotSet_0()
        allOderFilePath = '../data/citydata/season_1/training_data/order_data/temp/allorders.csv'
        df = pd.read_csv(allOderFilePath)
        df = df.loc[df['time_slotid'].isin(testSlots)]
        self.saveResultCsv(df, 'actual_0.csv')
        return
    
    
class Evaluate(GenerateResultCsv):
    def __init__(self):
        GenerateResultCsv.__init__(self)
        return
    def loadResultFiles(self, testSetNum):

        actualFile = 'actual_' + str(testSetNum) + '.csv'
        predictionFile = 'prediction_' + str(testSetNum) + '.csv'
        self.actualDict = self.loadDict(actualFile)
        self.predictonDict = self.loadDict(predictionFile)
        return
    
    def loadDict(self, filename):
        df = pd.read_csv(filename, header=None)
        res = {}
        for _, row in df.iterrows():
            res[(row[0], row[1])] = row[2]
        return res
    def calFinalResult(self, testSetNum):
        self.loadResultFiles(testSetNum)
        res = []
        for key, value in self.actualDict.iteritems():
            actual = value
            if actual == 0:
                print "record {} is 0, not included in final calculation".format(key)
                continue
            prediction = self.predictonDict[key]
            temp = (actual - prediction)/float(actual)
            if math.isnan(temp):
                print temp
            res.append(abs(temp))
        res = np.array(res)
        pd.DataFrame(res).to_csv('result.csv')
        print "final result: {}".format(res.mean())
        return np.mean(res)
    
    def run(self):
        self.generateActual_0()
        self.generatePrediction_0()
        self.calFinalResult(0)
        return
    


if __name__ == "__main__":   
    obj= Evaluate()
    obj.run()