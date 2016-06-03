from order import ExploreOrder
import pandas as pd
from utility.datafilepath import g_singletonDataFilePath
from sklearn.cross_validation import train_test_split
from timeslot import singletonTimeslot
from time import time
from utility.dumpload import DumpLoad
from sklearn import preprocessing
from enum import Enum

class ScaleMethod(Enum):
    NONE = 1
    MIN_MAX = 2
    STD = 3
    
class PrepareData(ExploreOrder):
    def __init__(self):
        ExploreOrder.__init__(self)
        self.count = 1
        self.dataDir = g_singletonDataFilePath.getOrderDir_Train()
        self.scaling = ScaleMethod.NONE
        self.usedFeatures = []
        self.excludeZerosActual = False
       
        return
    def getAllFeaturesDict(self):
        featureDict ={}
        preGaps = ['gap1', 'gap2', 'gap3']
        districtids = ['start_district_id_' + str(i + 1) for i in range(66)]
        timeids = ['time_id_' + str(i + 1) for i in range(144)]
        featureDict[1] = preGaps
        featureDict[2] = districtids
        featureDict[3] = timeids
        return featureDict
    def specifyUsedFeatures(self):
        if len(self.usedFeatures) == 0:
            unused = ['start_district_id', 'time_slotid', 'time_slot', 'all_requests', 'time_id']
            self.usedFeatures = [col for col in self.X_y_Df.columns if col not in ['gap']] 
            self.usedFeatures = [x for x in self.usedFeatures if x not in unused]
            return
        res = []
        featureDict = self.getAllFeaturesDict()
        [res.extend(featureDict[fea]) for fea in self.usedFeatures]
        self.usedFeatures = res
        return
    def loadRawData(self):
        gapDf, self.gapDict = self.loadGapData(self.dataDir + g_singletonDataFilePath.getGapFilename())
        self.X_y_Df = gapDf
        return
    def splitTrainTestSet(self):
        # Remove zeros values from data to try things out
        if self.excludeZerosActual:
            bNonZeros =   self.X_y_Df['gap'] != 0 
            self.X_y_Df = self.X_y_Df[bNonZeros]
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X_y_Df[self.usedFeatures], self.X_y_Df['gap'], test_size=0.25, random_state=42)
        return
    def rescaleFeatures(self):
        self.rescale(self.X_train)
        self.rescale(self.X_test)
        return
    def rescale(self, outX):
        scaler = None
        if self.scaling is ScaleMethod.STD:
            scaler = preprocessing.StandardScaler()
        elif self.scaling is ScaleMethod.MIN_MAX:
            scaler = preprocessing.MinMaxScaler()
        else:
            return outX
        outX[['gap1', 'gap2', 'gap3']] = scaler.fit_transform(outX[['gap1', 'gap2', 'gap3']])
        return outX
    def transformCategories(self):
        cols = ['start_district_id', 'time_id']
        for col in cols:
            col_data = pd.get_dummies(self.X_y_Df[col], prefix= col)
            self.X_y_Df = pd.concat([self.X_y_Df, col_data],  axis=1)
        return
    def transformPreGaps(self):
        t0 = time()
        dumpload = DumpLoad(self.dataDir + g_singletonDataFilePath.getPrevGapFileName())
        if dumpload.isExisiting():
            prevGaps = dumpload.load()
        else:
            prevGaps = self.X_y_Df.apply(self.getPrevGapsbyRow, axis = 1, raw=False, preNum = 3)
            dumpload.dump(prevGaps)
        self.X_y_Df = pd.concat([self.X_y_Df, prevGaps],  axis=1)
#         self.X_y_Df.to_csv('./temp/addprevgap.csv')
        print "transformPreGaps:", round(time()-t0, 3), "s"
#         print "prev gaps:\n", prevGaps.describe()
        
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
#     def removeUnusedCol(self):
#         self.X_y_Df.drop(['start_district_id', 'time_slotid', 'time_slot', 'all_requests', 'time_id'], axis=1, inplace=True)
#         return
    def loadTransformedData(self):
        self.loadRawData()
        self.transformPreGaps()
        self.transformCategories()
        self.specifyUsedFeatures()
#         self.X_y_Df.to_csv("temp/transformeddata.csv")
        return
    def getTrainTestSet(self):
        self.loadTransformedData()
        self.splitTrainTestSet()
        self.rescaleFeatures()
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
#         self.X_y_Df.to_csv('./temp/afterdummy.csv')
#         self.splitTrainTestSet()
        return
    


if __name__ == "__main__":   
    obj= PrepareData()
    obj.run()