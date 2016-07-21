import sys
import os
sys.path.insert(0, os.path.abspath('..')) 
# from pprint import pprint as p
# p(sys.path)


from exploredata.order import ExploreOrder
from exploredata.traffic import ExploreTraffic
from exploredata.weather import ExploreWeather
from prepareholdoutset import PrepareHoldoutSet
from utility.datafilepath import g_singletonDataFilePath
from utility.dumpload import DumpLoad
import numpy as np
import pandas as pd
from splittrainvalidation import SplitTrainValidation
from splittrainvalidation import HoldoutSplitMethod
from preprocess.historicaldata import HistoricalData
from preparegapcsv import prepareGapCsvForPrediction
from sklearn.preprocessing import LabelEncoder



    


    
class PrepareData(ExploreOrder, ExploreWeather, ExploreTraffic, PrepareHoldoutSet, SplitTrainValidation,HistoricalData, prepareGapCsvForPrediction):
    def __init__(self):
        ExploreOrder.__init__(self)
        self.usedFeatures = [101,102,103,2, 4,6, 701,702,703,801,802,10,11,1201,1202,1203,1204,1205,1206]
#         self.override_used_features = ['gap1', 'time_id', 'gap2', 'gap3', 'traffic2', 'traffic1', 'traffic3',
#                                        'preweather', 'start_district_id_28', 'start_district_id_8',
#                                        'start_district_id_7', 'start_district_id_48']
        self.usedLabel = 'gap'
        self.excludeZerosActual = True
        # the resultant data dictionary after preprocessing
        self.res_data_dict = {}
#         self.randomSate = None
#         self.test_size = 0.25
        self.holdout_split = HoldoutSplitMethod.IMITTATE_TEST2_PLUS2
        self.__label_encoder_dict = {}
       
        return
    def getAllFeaturesDict(self):
        featureDict ={}
#         preGaps = ['gap1', 'gap2', 'gap3']
        districtids = ['start_district_id_' + str(i + 1) for i in range(66)]
        timeids = ['time_id_' + str(i + 1) for i in range(144)]
        featureDict[101] = ['gap1']
        featureDict[102] = ['gap2']
        featureDict[103] = ['gap3']
        
        featureDict[2] = districtids
        featureDict[3] = timeids
        
        featureDict[4] = ['time_id']
        featureDict[5] = ['start_district_id']
        featureDict[6] = ['preweather']
        
        featureDict[701] = ['traffic1']
        featureDict[702] = ['traffic2']
        featureDict[703] = ['traffic3']
        
        
        featureDict[801] = ['gap_diff1']
        featureDict[802] = ['gap_diff2']
        
#         featureDict[901] = ['mean']
#         featureDict[902] = ['median']
#         featureDict[903] = ['plus_mean']
#         featureDict[904] = ['plus_median']
        
        featureDict[10] = ['district_gap_sum']
        featureDict[11] = ["rain_check"]
        
        featureDict[1201] = ['history_mean']
        featureDict[1202] = ['history_median']
        featureDict[1203] = ['history_mode']
        featureDict[1204] = ['history_plus_mean']
        featureDict[1205] = ['history_plus_median']
        featureDict[1206] = ['history_plus_mode']
        
        
        return featureDict
    def translateUsedFeatures(self):
        if  hasattr(self, 'override_used_features'):
            self.usedFeatures = self.override_used_features
            return
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
    def transformCategories(self):
#         cols = ['start_district_id', 'time_id']
        cols = ['start_district_id']
        self.X_y_Df['start_district_id'] = self.X_y_Df['start_district_id'].astype('category',categories=(np.arange(66) + 1))
#         self.X_y_Df['time_id'] = self.X_y_Df['time_id'].astype('category',categories=(np.arange(144) + 1))
        for col in cols:
            col_data = pd.get_dummies(self.X_y_Df[col], prefix= col)
            self.X_y_Df = pd.concat([self.X_y_Df, col_data],  axis=1)
        return
    def add_pre_gaps(self, data_dir):
        dumpfile_path = '../data_preprocessed/' + data_dir.split('/')[-2] + '_prevgap.df.pickle'
        dumpload = DumpLoad(dumpfile_path)
        if dumpload.isExisiting():
            df = dumpload.load()
        else:
            gap_dict = self.get_gap_dict(data_dir)
            df = self.X_y_Df[['start_district_id', 'time_slotid']].apply(self.find_prev_gap, axis = 1, pre_num = 3, gap_dict = gap_dict)
            dumpload.dump(df)
        self.X_y_Df = pd.concat([self.X_y_Df, df],  axis=1)
        return
#     def add_gap_mean_median(self, data_dir):
#         dumpload = DumpLoad(data_dir + 'order_data/temp/gapmeanmedian.df.pickle')
#         if dumpload.isExisiting():
#             df = dumpload.load()
#         else:
#             temp_dict = self.get_gap_meanmedian_dict()
#             df = self.X_y_Df[['start_district_id', 'time_id']].apply(self.find_gap_meanmedian, axis = 1, gap_meanmedian_dict = temp_dict)
#             dumpload.dump(df)
#         self.X_y_Df = pd.concat([self.X_y_Df, df],  axis=1)
#         return
    def add_rain_check(self):
        rain_dict ={1:1, 2:0,3:0,4:0,5:0,6:0,7:0,8:0,9:0}
        self.X_y_Df["rain_check"] = self.X_y_Df["preweather"].map(rain_dict)
        return
    def add_prev_weather(self, data_dir):
        dumpfile_path = '../data_preprocessed/' + data_dir.split('/')[-2] + '_prevweather.df.pickle'
        dumpload = DumpLoad(dumpfile_path)
        if dumpload.isExisiting():
            df = dumpload.load()
        else:
            weather_dict = self.get_weather_dict(data_dir)
            
            df = self.X_y_Df['time_slotid'].apply(self.find_prev_weather_mode, weather_dict=weather_dict)
                    
            dumpload.dump(df)
        self.X_y_Df = pd.concat([self.X_y_Df, df],  axis=1)
        self.add_rain_check()
        return
    def add_prev_traffic(self, data_dir):
        dumpfile_path = '../data_preprocessed/' + data_dir.split('/')[-2] + '_prevtraffic.df.pickle'
        dumpload = DumpLoad(dumpfile_path)
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
    def add_gap_difference(self):
        self.X_y_Df['gap_diff1'] = self.X_y_Df['gap2'] - self.X_y_Df['gap1']
        self.X_y_Df['gap_diff2'] = self.X_y_Df['gap3'] - self.X_y_Df['gap2']
        return
    def add_district_gap_sum(self):
        dumpfile_path = '../data_preprocessed/' +'training_data_district_gap_sum.dict.pickle'
        dumpload = DumpLoad(dumpfile_path)
        if dumpload.isExisiting():
            district_gap_sum_dict = dumpload.load()
        else:
            district_gap_sum_dict = self.X_y_Df.groupby('start_district_id')['gap'].sum().to_dict()
            dumpload.dump(district_gap_sum_dict)
            
        self.X_y_Df["district_gap_sum"] = self.X_y_Df["start_district_id"].map(district_gap_sum_dict)
        return
    def add_history_data(self,data_dir):
        dumpfile_path = '../data_preprocessed/' + data_dir.split('/')[-2] + '_history_data.df.pickle'
        dumpload = DumpLoad(dumpfile_path)
        if dumpload.isExisiting():
            df = dumpload.load()
        else:
            temp_dict = self.get_history_data_dict()
            df = self.X_y_Df[['start_district_id', 'time_id']].apply(self.find_history_data, axis = 1, history_dict = temp_dict)
            dumpload.dump(df)
        self.X_y_Df = pd.concat([self.X_y_Df, df],  axis=1)
        return
    def __add_cross_features(self, data_dir):
        self.__add_cross_features_(['start_district_id', 'time_id'], 'time_district', data_dir)
        return
    def __add_cross_features_(self, exising_feature_names, new_feature_name):
        
        for i in range(len(exising_feature_names)):
            if i ==0:
                self.X_y_Df[new_feature_name] = self.X_y_Df[exising_feature_names[i]].astype(str)
                continue
            
            self.X_y_Df[new_feature_name] = self.X_y_Df[new_feature_name] + '_' + self.X_y_Df[exising_feature_names[i]].astype(str)  
         
#         self.X_y_Df[new_feature_name]  = self.X_y_Df[exising_feature_names].apply(lambda x: '_'.join(x.astype(str)), axis=1)
        
#         if data_dir == g_singletonDataFilePath.getTrainDir():
#             el = LabelEncoder()
#             el.fit(self.X_y_Df[new_feature_name])
#             self.__label_encoder_dict[new_feature_name] = el
#         
#         el = self.__label_encoder_dict[new_feature_name]
#         self.X_y_Df[new_feature_name] = el.transform(self.X_y_Df[new_feature_name])
        return
    def __engineer_feature(self, data_dir = None):
        self.add_pre_gaps(data_dir)
#         self.add_gap_mean_median(data_dir)
        self.add_district_gap_sum()
        self.add_prev_weather(data_dir)
        self.add_prev_traffic(data_dir)
        self.add_gap_difference()
        self.add_history_data(data_dir)
        self.remove_zero_gap()
        self.transformCategories()
        return

    def get_train_validationset(self, foldid = -1): 
        data_dir = g_singletonDataFilePath.getTrainDir()
        self.__do_prepare_data()
        df, cv = self.res_data_dict[data_dir]
        folds = []
        for train_index, test_index in cv:
            folds.append((train_index, test_index))
        train_index = folds[foldid][0]
        test_index = folds[foldid][1]
        self.X_train = df.iloc[train_index][self.usedFeatures]
        self.y_train = df.iloc[train_index][self.usedLabel]     
        self.X_test =  df.iloc[test_index][self.usedFeatures]
        self.y_test =  df.iloc[test_index][self.usedLabel]
        return   
    def getFeaturesLabel(self):
        data_dir = g_singletonDataFilePath.getTrainDir()
        self.__do_prepare_data()
        df, cv = self.res_data_dict[data_dir]
        return df[self.usedFeatures], df[self.usedLabel],cv
        
        
        return
    def __get_feature_label(self):
        data_dir = g_singletonDataFilePath.getTrainDir()
        self.X_y_Df = self.load_gapdf(data_dir) 
        self.__engineer_feature(data_dir)

        if self.holdout_split == HoldoutSplitMethod.kFOLD_FORWARD_CHAINING:
            cv = self.kfold_forward_chaining(self.X_y_Df)
        elif self.holdout_split == HoldoutSplitMethod.KFOLD_BYDATE:
            cv = self.kfold_bydate(self.X_y_Df)
        else:
            cv = self.get_imitate_testset2(self.X_y_Df, split_method = self.holdout_split)

        self.res_data_dict[data_dir] = self.X_y_Df,cv
        return
    def __get_feature_for_test_set(self,data_dir):
        self.X_y_Df = self.load_prediction_csv(data_dir)
        self.__engineer_feature(data_dir)
        self.res_data_dict[data_dir] = self.X_y_Df
        return
    
    def getFeaturesforTestSet(self, data_dir):
        self.__do_prepare_data()
        return self.res_data_dict[data_dir]
        
    def __do_prepare_data(self):
        if len(self.res_data_dict) != 0:
            # the data has already been preprocessed
            return
        self.__get_feature_label()
        self.__get_feature_for_test_set(g_singletonDataFilePath.getTest2Dir())
        self.__get_feature_for_test_set(g_singletonDataFilePath.getTest1Dir())
        
        self.translateUsedFeatures()

        return
    def run(self):
        self.getFeaturesLabel()
        self.getFeaturesforTestSet(g_singletonDataFilePath.getTest2Dir())
        self.getFeaturesforTestSet(g_singletonDataFilePath.getTest1Dir())


        return
    


if __name__ == "__main__":   
    obj= PrepareData()
    obj.run()