import sys
import os
sys.path.insert(0, os.path.abspath('..')) 
# from pprint import pprint as p
# p(sys.path)

from enum import Enum
from exploredata.order import ExploreOrder
from exploredata.traffic import ExploreTraffic
from exploredata.weather import ExploreWeather
from prepareholdoutset import PrepareHoldoutSet
from utility.datafilepath import g_singletonDataFilePath
from utility.dumpload import DumpLoad
import numpy as np
import pandas as pd



    
class HoldoutSplitMethod(Enum):
#     NONE = 1
    BYDATESLOT_RANDOM = 2
    IMITATE_PUBLICSET = 3

    
class PrepareData(ExploreOrder, ExploreWeather, ExploreTraffic, PrepareHoldoutSet):
    def __init__(self):
        ExploreOrder.__init__(self)
        self.usedFeatures = [1,2,3,6,7]
        self.usedLabel = 'gap'
        self.excludeZerosActual = True
        self.randomSate = None
        self.test_size = 0.25
        self.holdout_split = HoldoutSplitMethod.IMITATE_PUBLICSET
       
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
        featureDict[6] = ['preweather']
        featureDict[7] = ['traffic1','traffic2','traffic3']
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
    def splitTrainTestSet(self):
        # Remove zeros values from data to try things out 
#         if self.holdout_split == HoldoutSplitMethod.NONE:     
#             self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X_y_Df[self.usedFeatures], self.X_y_Df['gap'], test_size=self.test_size, random_state=self.randomSate)
#             return
        if self.holdout_split == HoldoutSplitMethod.BYDATESLOT_RANDOM: 
            self.splitby_random_dateslot()
            return
        self.splitby_imitate_publicset()
        return
    def splitby_imitate_publicset(self):
        validation_dateslots = self.get_holdoutset(holdout_id = 1)
        validation_dateslots = self.X_y_Df['time_slotid'].isin(validation_dateslots)
        train_dateslots = self.X_y_Df['time_date'] < '2016-01-13'
        
        self.dateslot_test_num = self.X_y_Df[validation_dateslots]['time_slotid'].unique().shape[0]
        self.dateslot_train_num = self.X_y_Df[train_dateslots]['time_slotid'].unique().shape[0]
        
        
        self.X_test = self.X_y_Df[validation_dateslots][self.usedFeatures]
        self.y_test = self.X_y_Df[validation_dateslots][self.usedLabel]
        
        self.X_train = self.X_y_Df[train_dateslots][self.usedFeatures]
        self.y_train = self.X_y_Df[train_dateslots][self.usedLabel]    
        return
    def splitby_dateslots(self, selected_dateslots):
        selected_dateslots = self.X_y_Df['time_slotid'].isin(selected_dateslots)
        self.X_test = self.X_y_Df[selected_dateslots][self.usedFeatures]
        self.y_test = self.X_y_Df[selected_dateslots][self.usedLabel]
        self.X_train = self.X_y_Df[np.logical_not(selected_dateslots)][self.usedFeatures]
        self.y_train = self.X_y_Df[np.logical_not(selected_dateslots)][self.usedLabel]
        return
    def splitby_random_dateslot(self):
        all_dateslots = self.X_y_Df['time_slotid'].unique()
        self.dateslot_test_num = int(self.test_size *all_dateslots.shape[0])
        self.dateslot_train_num = all_dateslots.shape[0] - self.dateslot_test_num
        
        selected_dateslots = np.random.choice(all_dateslots,  size=self.dateslot_test_num, replace=False)
        self.splitby_dateslots(selected_dateslots)
        
        return
    def transformCategories(self):
        cols = ['start_district_id', 'time_id']
        self.X_y_Df['start_district_id'] = self.X_y_Df['start_district_id'].astype('category',categories=(np.arange(66) + 1))
        self.X_y_Df['time_id'] = self.X_y_Df['time_id'].astype('category',categories=(np.arange(144) + 1))
        for col in cols:
            col_data = pd.get_dummies(self.X_y_Df[col], prefix= col)
            self.X_y_Df = pd.concat([self.X_y_Df, col_data],  axis=1)
        print self.X_y_Df.columns
        return
    def add_pre_gaps(self, data_dir):
        dumpload = DumpLoad(data_dir + 'order_data/temp/prevgap.df.pickle')
        if dumpload.isExisiting():
            df = dumpload.load()
        else:
            gap_dict = self.get_gap_dict(data_dir)
            df = self.X_y_Df[['start_district_id', 'time_slotid']].apply(self.find_prev_gap, axis = 1, pre_num = 3, gap_dict = gap_dict)
            dumpload.dump(df)
        self.X_y_Df = pd.concat([self.X_y_Df, df],  axis=1)
        return
    def add_prev_weather(self, data_dir):
        dumpload = DumpLoad(data_dir + 'weather_data/temp/prevweather.df.pickle')
        if dumpload.isExisiting():
            df = dumpload.load()
        else:
            weather_dict = self.get_weather_dict(data_dir)
            
            df = self.X_y_Df['time_slotid'].apply(self.find_prev_weather, weather_dict=weather_dict)
                    
            dumpload.dump(df)
        self.X_y_Df = pd.concat([self.X_y_Df, df],  axis=1)
        return
    def add_prev_traffic(self, data_dir):
        dumpload = DumpLoad(data_dir + 'traffic_data/temp/prevtraffic.df.pickle')
        if dumpload.isExisiting():
            df = dumpload.load()
        else:
            traffic_dict = self.get_traffic_dict(data_dir)
            
            df = self.X_y_Df[['start_district_id', 'time_slotid']].apply(self.find_prev_traffic,axis = 1, traffic_dict=traffic_dict, pre_num = 3)
            
            dumpload.dump(df)
        self.X_y_Df = pd.concat([self.X_y_Df, df],  axis=1)
        return
    def remove_zero_gap(self):
        if not 'gap' in self.X_y_Df.columns:
            # when we perform validation on test set, we do not expect to have 'gap' column
            return
        if self.excludeZerosActual:
            bNonZeros =   self.X_y_Df['gap'] != 0 
            self.X_y_Df = self.X_y_Df[bNonZeros]
        return
    def transformXfDf(self, data_dir = None):
        self.add_pre_gaps(data_dir)
        self.add_prev_weather(data_dir)
        self.add_prev_traffic(data_dir)
        self.remove_zero_gap()
#         self.transformCategories()
        if hasattr(self, 'busedFeaturesTranslated'):
            return
        self.translateUsedFeatures()
        self.busedFeaturesTranslated = True
#         self.X_y_Df.to_csv("temp/transformeddata.csv")
        return
    def getTrainTestSet(self):
        data_dir = g_singletonDataFilePath.getTrainDir()
        self.X_y_Df = self.load_gapdf(data_dir)
        self.transformXfDf(data_dir)
        
        self.splitTrainTestSet()
        return (self.X_train, self.X_test, self.y_train, self.y_test)
        
    def getFeaturesLabel(self):
        data_dir = g_singletonDataFilePath.getTrainDir()
        self.X_y_Df = self.load_gapdf(data_dir) 
        self.transformXfDf(data_dir)
#         self.remove_zero_gap()
         
        return self.X_y_Df[self.usedFeatures], self.X_y_Df[self.usedLabel]
    def getFeaturesforTestSet(self, data_dir):
        self.X_y_Df = pd.read_csv(data_dir + 'gap_prediction.csv', index_col=0)
        self.transformXfDf(data_dir)
        return self.X_y_Df
    def run(self):
#         print self.getFeaturesforTestSet(g_singletonDataFilePath.getTest1Dir())
        
        
        self.getTrainTestSet()
#         self.getTrainTestSet()
#         self.getFeaturesLabel()
#         self.getFeaturesforTestSet(g_singletonDataFilePath.getTest1Dir())

        return
    


if __name__ == "__main__":   
    obj= PrepareData()
    obj.run()