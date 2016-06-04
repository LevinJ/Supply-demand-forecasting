import sys
import os
sys.path.insert(0, os.path.abspath('..')) 
# from pprint import pprint as p
# p(sys.path)

from exploredata.order import ExploreOrder
import pandas as pd
from utility.datafilepath import g_singletonDataFilePath
from sklearn.cross_validation import train_test_split
from exploredata.timeslot import singletonTimeslot
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
        self.scaling = ScaleMethod.NONE
        self.usedFeatures = []
        self.usedLabel = 'gap'
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
        featureDict[4] = ['time_id']
        featureDict[5] = ['start_district_id']
        return featureDict
    def translateUsedFeatures(self):
        if len(self.usedFeatures) == 0:
            unused = ['time_slotid', 'time_slot', 'all_requests']
#             unused = ['start_district_id', 'time_slotid', 'time_slot', 'all_requests', 'time_id']
            self.usedFeatures = [col for col in self.X_y_Df.columns if col not in ['gap']] 
            self.usedFeatures = [x for x in self.usedFeatures if x not in unused]
            return
        res = []
        featureDict = self.getAllFeaturesDict()
        [res.extend(featureDict[fea]) for fea in self.usedFeatures]
        self.usedFeatures = res
        return
#     def loadRawData(self, dataDir):
#         gapDf, self.gapDict = self.loadGapData(dataDir + g_singletonDataFilePath.getGapFilename())
#         self.X_y_Df = gapDf
#         return
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
        print self.X_y_Df.columns
        return
    def addPreGaps(self, prevGapDfDumpDir):
#         t0 = time()
        dumpload = DumpLoad(prevGapDfDumpDir + 'prevgap.df.pickle')
        if dumpload.isExisiting():
            prevGaps = dumpload.load()
        else:
            self.gapDict = self.loadGapDict(prevGapDfDumpDir + 'gap.csv.dict.pickle')
            prevGaps = self.X_y_Df.apply(self.getPrevGapsbyRow, axis = 1, raw=False, preNum = 3)
            dumpload.dump(prevGaps)
        self.X_y_Df = pd.concat([self.X_y_Df, prevGaps],  axis=1)
        
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
    def transformXfDf(self, prevGapDfDumpDir = None):
        self.addPreGaps(prevGapDfDumpDir)
#         self.transformCategories()
        if hasattr(self, 'busedFeaturesTranslated'):
            return
        self.translateUsedFeatures()
        self.busedFeaturesTranslated = True
#         self.X_y_Df.to_csv("temp/transformeddata.csv")
        return
    def getTrainTestSet(self):
        dataDir = g_singletonDataFilePath.getTrainDir()
        self.X_y_Df = self.loadGapCsvFile(dataDir + 'order_data/temp/gap.csv')
        self.transformXfDf(prevGapDfDumpDir = dataDir + 'order_data/temp/')
        
        self.splitTrainTestSet()
        self.rescaleFeatures()
        return (self.X_train, self.X_test, self.y_train, self.y_test)
        
    def getFeaturesLabel(self):
        dataDir = g_singletonDataFilePath.getTrainDir()
        self.X_y_Df = self.loadGapCsvFile(dataDir + 'order_data/temp/gap.csv') 
        self.transformXfDf(prevGapDfDumpDir = dataDir + 'order_data/temp/')
         
        return self.X_y_Df[self.usedFeatures], self.X_y_Df[self.usedLabel]
    def getFeaturesforTestSet(self, dataDir):
        self.X_y_Df = pd.read_csv(dataDir + 'gap_prediction.csv', index_col=0)
        self.transformXfDf(prevGapDfDumpDir = dataDir + 'order_data/temp/')
        return self.X_y_Df
    def run(self):
        print self.getFeaturesforTestSet(g_singletonDataFilePath.getTest1Dir())
        
        
#         self.getTrainTestSet()
#         self.getFeaturesLabel()

        return
    


if __name__ == "__main__":   
    obj= PrepareData()
    obj.run()