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
from sklearn.preprocessing import OneHotEncoder
from exploredata.poi import ExplorePoi



    


    
class PrepareData(ExploreOrder, ExploreWeather, ExploreTraffic, PrepareHoldoutSet, SplitTrainValidation,HistoricalData, prepareGapCsvForPrediction,ExplorePoi):
    def __init__(self):
        ExploreOrder.__init__(self)
#         self.usedFeatures = []
#         self.usedFeatures = [101,102,103,104,105,106,107, 
#                              201, 202, 203,204,205,206,
#                              301, 302,
#                              401,402,
#                              501,502,503,504,505,506,507,
#                              601,602,603,604,605,606,
#                              8801,8802
#                              ]
#         self.usedFeatures = [101,102,103]
        self.usedFeatures = [101,102,103,104,105,106,107, 
                             201, 203,204,205,206,
                             301,
                             401,402,
                             501,502,503,504,505,506,507,
                             601,602,603,604,605,606,
                             8801,8802
                             ]
#         self.override_used_features = ['gap1', 'time_id', 'gap2', 'gap3', 'traffic2', 'traffic1', 'traffic3',
#                                        'preweather', 'start_district_id_28', 'start_district_id_8',
#                                        'start_district_id_7', 'start_district_id_48']
        self.usedLabel = 'gap'
        self.excludeZerosActual = True
        # the resultant data dictionary after preprocessing
        self.res_data_dict = {}
#         self.randomSate = None
#         self.test_size = 0.25
#         self.holdout_split = HoldoutSplitMethod.IMITTATE_TEST2_FULL
#         self.holdout_split = HoldoutSplitMethod.KFOLD_BYDATE
        self.holdout_split = HoldoutSplitMethod.kFOLD_FORWARD_CHAINING
        self.train_validation_foldid = -1

       
        return
    
    def __get_all_features_dict(self):
        featureDict ={}
#         preGaps = ['gap1', 'gap2', 'gap3']
        districtids = ['start_district_id_' + str(i + 1) for i in range(66)]
        timeids = ['time_id_' + str(i + 1) for i in range(144)]
        
        
        # gap features
        featureDict[101] = ['gap1']
        featureDict[102] = ['gap2']
        featureDict[103] = ['gap3']
        featureDict[104] = ['gap_diff1']
        featureDict[105] = ['gap_diff2'] 
        featureDict[106] = ['gap_mean']
        featureDict[107] = ['gap_std']
        
        
        
        #district features
        featureDict[201] = ['start_district_id']
        featureDict[202] = districtids
        featureDict[203] = ['start_district_id_51', 'start_district_id_23','start_district_id_8','start_district_id_37']
        featureDict[204] = ['district_gap_sum']
        featureDict[205] = self.get_district_type_list()
        featureDict[206] = ['poi_sum']
        
        #time features
        featureDict[301] = ['time_id']
        featureDict[302] = timeids
        
        #weatehr features
        featureDict[401] = ['preweather']
        featureDict[402] = ["rain_check"]
        
        # Traffic features
        featureDict[501] = ['traffic1']
        featureDict[502] = ['traffic2']
        featureDict[503] = ['traffic3']
        featureDict[504] = ['traffic_diff1']
        featureDict[505] = ['traffic_diff2']
        featureDict[506] = ['traffic_mean']
        featureDict[507] = ['traffic_std']

        
        #historical features
        featureDict[601] = ['history_mean']
        featureDict[602] = ['history_median']
        featureDict[603] = ['history_mode']
        featureDict[604] = ['history_plus_mean']
        featureDict[605] = ['history_plus_median']
        featureDict[606] = ['history_plus_mode']
        
        #cross features
        featureDict[8801] = ['district_time']
        featureDict[8802] = ['weather_time']
        

        return featureDict
    def __get_label_encode_dict(self):
        le_dict = {}
        le_dict[('start_district_id', 'time_id')] = 'district_time'
        le_dict[('preweather', 'time_id')] = 'weather_time'
        le_dict[('time_id')] = 'time_id'
        le_dict[('start_district_id')] = 'start_district_id'
        
        return le_dict
#     def __translate_used_features(self):
#         if  hasattr(self, 'override_used_features'):
#             self.usedFeatures = self.override_used_features
#             return
# 
#         res = []
#         featureDict = self.__get_all_features_dict()
#         [res.extend(featureDict[fea]) for fea in self.usedFeatures]
#         
#         return res
    def get_used_features(self):
        res = []
        featureDict = self.__get_all_features_dict()
        [res.extend(featureDict[fea]) for fea in self.usedFeatures]
        return res

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
    
    def __add_poi(self, data_dir):
        dumpfile_path = '../data_preprocessed/' + data_dir.split('/')[-2] + '_poi.df.pickle'
        dumpload = DumpLoad(dumpfile_path)
        if dumpload.isExisiting():
            df = dumpload.load()
        else:
            poi_dict = self.get_district_type_dict()
            
            df = self.X_y_Df[['start_district_id']].apply(self.find_poi,axis = 1, poi_dict=poi_dict)
            
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
    def __add_gap_statistics(self):
        self.X_y_Df['gap_diff1'] = self.X_y_Df['gap2'] - self.X_y_Df['gap1']
        self.X_y_Df['gap_diff2'] = self.X_y_Df['gap3'] - self.X_y_Df['gap2']
        self.X_y_Df['gap_mean'] = self.X_y_Df[['gap1','gap2','gap3']].mean(axis=1)
        self.X_y_Df['gap_std'] = self.X_y_Df[['gap1','gap2','gap3']].std(axis=1)
        return
    def __add_traffic_statistics(self):
        self.X_y_Df['traffic_diff1'] = self.X_y_Df['traffic2'] - self.X_y_Df['traffic1']
        self.X_y_Df['traffic_diff2'] = self.X_y_Df['traffic3'] - self.X_y_Df['traffic2']
        self.X_y_Df['traffic_mean'] = self.X_y_Df[['traffic1','traffic2','traffic3']].mean(axis=1)
        self.X_y_Df['traffic_std'] = self.X_y_Df[['traffic1','traffic2','traffic3']].std(axis=1)
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
    def __add_cross_features(self):
        cross_feature_dict = self.__get_label_encode_dict()
        for exising_feature_names, new_feature_name in cross_feature_dict.iteritems():
            if isinstance(exising_feature_names, basestring):
                # such items in the dict only need to do label encoding, and not cross feature
                continue
            self.__add_cross_feature(exising_feature_names, new_feature_name)
        return
    def __add_cross_feature(self, exising_feature_names, new_feature_name):
        
        for i in range(len(exising_feature_names)):
            if i ==0:
                self.X_y_Df[new_feature_name] = self.X_y_Df[exising_feature_names[i]].astype(str)
                continue
            
            self.X_y_Df[new_feature_name] = self.X_y_Df[new_feature_name] + '_' + self.X_y_Df[exising_feature_names[i]].astype(str)  
        return
    def __add_poi_sum(self):
        self.X_y_Df['poi_sum'] = self.X_y_Df[self.get_district_type_list()].sum(axis = 1)
        return
    def __engineer_feature(self, data_dir = None):
        self.add_pre_gaps(data_dir)
        self.add_district_gap_sum()
        self.add_prev_weather(data_dir)
        self.add_prev_traffic(data_dir)
        self.__add_gap_statistics()
        self.__add_traffic_statistics()
        self.__add_poi(data_dir)
        self.__add_poi_sum()
        self.add_history_data(data_dir)
        self.remove_zero_gap()
        self.__add_cross_features()
        return

    def get_train_validationset(self):
        data_dir = g_singletonDataFilePath.getTrainDir()
        self.__do_prepare_data()
        df, cv = self.res_data_dict[data_dir]
        folds = []
        for train_index, test_index in cv:
            folds.append((train_index, test_index))
        train_index = folds[self.train_validation_foldid][0]
        test_index = folds[self.train_validation_foldid][1]
        X_train = df.iloc[train_index][self.get_used_features()]
        y_train = df.iloc[train_index][self.usedLabel]     
        X_test =  df.iloc[test_index][self.get_used_features()]
        y_test =  df.iloc[test_index][self.usedLabel]
        return  X_train, y_train,X_test,y_test
    def getFeaturesLabel(self):
        data_dir = g_singletonDataFilePath.getTrainDir()
        self.__do_prepare_data()
        df, cv = self.res_data_dict[data_dir]
        return df[self.get_used_features()], df[self.usedLabel],cv
        
        
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
     
    def __do_label_encoding(self):
        df_train, _ = self.res_data_dict[g_singletonDataFilePath.getTrainDir()]
        df_testset1 = self.res_data_dict[g_singletonDataFilePath.getTest1Dir()]
        df_testset2 = self.res_data_dict[g_singletonDataFilePath.getTest2Dir()]
        le = LabelEncoder()
        cross_feature_dict = self.__get_label_encode_dict()
        for _, new_feature_name in cross_feature_dict.iteritems():
            to_be_stacked = [df_train[new_feature_name], df_testset1[new_feature_name], df_testset2[new_feature_name]]
            le.fit(pd.concat(to_be_stacked, axis=0))
            df_train[new_feature_name] = le.transform(df_train[new_feature_name])
            df_testset1[new_feature_name] = le.transform(df_testset1[new_feature_name])
            df_testset2[new_feature_name] = le.transform(df_testset2[new_feature_name])
            
        return 
    def __save_final_data(self):
        df_train, _ = self.res_data_dict[g_singletonDataFilePath.getTrainDir()]
        df_testset1 = self.res_data_dict[g_singletonDataFilePath.getTest1Dir()]
        df_testset2 = self.res_data_dict[g_singletonDataFilePath.getTest2Dir()]
        df_train.to_csv('temp/df_train_final.csv')
        df_testset1.to_csv('temp/df_testset1_final.csv')
        df_testset2.to_csv('temp/df_testset2_final.csv')
        return
    def __get_expanded_col_names(self, cols, sub_cols):
        """
        helper method to generate expanded columns after one hot encoding
        cols, original column names ['a', 'b', 'c']
        sub_cols, one hot code lenght for each original column [2,3,4]
        res, the new column names, ['a_1', 'a_2', 'b_1', 'b_2', 'b_3', 'c_1', 'c_2', 'c_3', 'c_4']
        """
        res = []
        if len(cols) != len(sub_cols):
            raise "cols and expanded sub columns are not consistent"
        for i in range(len(cols)):
            prefix = cols[i]
            sub_num = sub_cols[i]
            for j in range(sub_num):
                res.append(prefix + '_' + str(j + 1))
        return res
    def __filter_too_big_onehot_encoding(self, enc, to_be_encoded_old, df_train, df_testset1, df_testset2):
        print "Filter out too big one hot encoding (>=200)", np.array(to_be_encoded_old)[enc.n_values_ >= 200]
        to_be_encoded = np.array(to_be_encoded_old)[enc.n_values_ < 200]
        to_be_stacked_df = pd.concat([df_train[to_be_encoded], df_testset1[to_be_encoded], df_testset2[to_be_encoded]], axis = 0)
        enc.fit(to_be_stacked_df)
        return enc, to_be_encoded
    def __do_one_hot_encodings(self):
        df_train, cv = self.res_data_dict[g_singletonDataFilePath.getTrainDir()]
        df_testset1 = self.res_data_dict[g_singletonDataFilePath.getTest1Dir()]
        df_testset2 = self.res_data_dict[g_singletonDataFilePath.getTest2Dir()]
        enc = OneHotEncoder(sparse=False)
        cross_feature_dict = self.__get_label_encode_dict()
        to_be_encoded = []
        for _, new_feature_name in cross_feature_dict.iteritems():
            to_be_encoded.append(new_feature_name)
        #fix all data source
        to_be_stacked_df = pd.concat([df_train[to_be_encoded], df_testset1[to_be_encoded], df_testset2[to_be_encoded]], axis = 0)
        enc.fit(to_be_stacked_df)
        
        enc, to_be_encoded = self.__filter_too_big_onehot_encoding(enc, to_be_encoded, df_train, df_testset1, df_testset2)
        # transform on seprate data source
        self.res_data_dict[g_singletonDataFilePath.getTrainDir()] = self.__do_one_hot_encoding(df_train, enc, to_be_encoded),cv
        self.res_data_dict[g_singletonDataFilePath.getTest1Dir()] = self.__do_one_hot_encoding(df_testset1,enc, to_be_encoded)
        self.res_data_dict[g_singletonDataFilePath.getTest2Dir()] = self.__do_one_hot_encoding(df_testset2, enc, to_be_encoded)
        return
    def __do_one_hot_encoding(self, df, enc, to_be_encoded):
        arr = enc.transform(df[to_be_encoded])
        
        new_col_names = self.__get_expanded_col_names(to_be_encoded, enc.n_values_)
        df_res = pd.DataFrame(arr, columns=new_col_names)
        df = pd.concat([df, df_res], axis = 1)
        return df
   
    def __do_prepare_data(self):
        if len(self.res_data_dict) != 0:
            # the data has already been preprocessed
            return
        self.__get_feature_label()
        self.__get_feature_for_test_set(g_singletonDataFilePath.getTest2Dir())
        self.__get_feature_for_test_set(g_singletonDataFilePath.getTest1Dir())
        self.__do_label_encoding()
        self.__do_one_hot_encodings()

        return
    def run(self):
        self.getFeaturesLabel()
        self.getFeaturesforTestSet(g_singletonDataFilePath.getTest2Dir())
        self.getFeaturesforTestSet(g_singletonDataFilePath.getTest1Dir())
        self.get_train_validationset()
#         self.__save_final_data()


        return
    


if __name__ == "__main__":   
    obj= PrepareData()
    obj.run()