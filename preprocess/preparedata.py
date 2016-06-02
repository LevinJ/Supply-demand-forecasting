from order import ExploreOrder
import pandas as pd
from utility.datafilepath import g_singletonDataFilePath
from sklearn.cross_validation import train_test_split
from timeslot import singletonTimeslot
from time import time
from utility.dumpload import DumpLoad


class PrepareData(ExploreOrder):
    def __init__(self):
        ExploreOrder.__init__(self)
        self.count = 1
        self.dataDir = g_singletonDataFilePath.getOrderDir_Train()
       
        return
    def loadRawData(self):
        self.gapDf, self.gapDict = self.loadGapData(self.dataDir + g_singletonDataFilePath.getGapFilename())
        return
    def splitTrainTestSet(self):
        xcols = [col for col in self.gapDf.columns if col not in ['gap']]     
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.gapDf[xcols], self.gapDf['gap'], test_size=0.25, random_state=42)
        return
    def transformCategories(self):
        col_data = pd.get_dummies(self.gapDf['start_district_id'], prefix='start_district_')
        self.gapDf = pd.concat([self.gapDf, col_data],  axis=1)
        return
    def transformPreGaps(self):
        t0 = time()
        dumpload = DumpLoad(self.dataDir + g_singletonDataFilePath.getPrevGapFileName())
        if dumpload.isExisiting():
            prevGaps = dumpload.load()
        else:
            prevGaps = self.gapDf.apply(self.getPrevGapsbyRow, axis = 1, raw=False, preNum = 3)
            dumpload.dump(prevGaps)
        self.gapDf = pd.concat([self.gapDf, prevGaps],  axis=1)
#         self.gapDf.to_csv('./temp/addprevgap.csv')
        print "transformPreGaps:", round(time()-t0, 3), "s"
        print "prev gaps:\n", prevGaps.describe()
        
        return
    def getPrevGapsbyRow(self, row, preNum = 3):
        start_district_id = row['start_district_id']
        time_slotid = row['time_slotid'] 
        index = ['gap' + str(i + 1) for i in range(preNum)]
        res =pd.Series(self.getPrevGaps(start_district_id, time_slotid, preNum), index = index)
        return res
    def getPrevGaps(self, start_district_id, time_slotid,preNum):
        res = []
        prevSlots = singletonTimeslot.getPrevSlots(time_slotid, preNum)
        for prevslot in prevSlots:
            try:
                res.append(self.gapDict[(start_district_id, prevslot)])
            except:
                res.append(0)
        return res
    def unitTest(self):
        assert [3096,1698,318,33,0,0] == self.getPrevGaps(51, '2016-01-01-5', 6)
        assert [0,0,0] == self.getPrevGaps(45, '2016-01-16-2', 3)
        assert [24,26,37] == self.getPrevGaps(53, '2016-01-04-56', 3)
        print "unit test passed"
        return
    def removeUnusedCol(self):
        self.gapDf.drop(['start_district_id', 'time_slotid', 'time_slot', 'all_requests'], axis=1, inplace=True)
        return
    def getTrainTestSet(self):
        self.loadRawData()
        self.transformPreGaps()
        self.transformCategories()
        self.removeUnusedCol()
        self.splitTrainTestSet()
        return (self.X_train, self.X_test, self.y_train, self.y_test)
        
        
        return
    def run(self):
        self.getTrainTestSet()
#         self.loadRawData()
#         self.unitTest()       
#         
#         self.transformPreGaps()
#         self.transformCategories()
#         
#         self.gapDf.to_csv('./temp/afterdummy.csv')
#         self.splitTrainTestSet()
        return
    


if __name__ == "__main__":   
    obj= PrepareData()
    obj.run()