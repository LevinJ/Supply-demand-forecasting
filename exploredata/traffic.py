from utility.datafilepath import g_singletonDataFilePath
from timeslot import singletonTimeslot
import pandas as pd
import os
from exploredata import ExploreData
from time import time
from utility.dumpload import DumpLoad
from  districtid import singletonDistricId


class ExploreTraffic(ExploreData ):
    def __init__(self):
        return
    def run(self):
#         self.__unittest()
        data_dir = g_singletonDataFilePath.getTest2Dir()
#         self.save_all_csv(data_dir+ 'traffic_data/')
#         self.combine_all_csv(data_dir + 'traffic_data/temp/', 'traffic_', 'traffic.csv')
        self.get_traffic_dict(data_dir)
        return
    def __unittest(self):
        #         self.combine_all_csv(g_singletonDataFilePath.getTrainDir() + 'weather_data/temp/', 'weather_', 'weather.csv')
#         self.save_one_csv(g_singletonDataFilePath.getTrainDir() + 'traffic_data/traffic_data_2016-01-04')
#         weatherdf = self.load_weatherdf(g_singletonDataFilePath.getTrainDir())
        data_dir = g_singletonDataFilePath.getTrainDir()
        traffic_dict = self.get_traffic_dict(data_dir)
        assert [0,0,0] == self.find_prev_traffic(pd.Series([1, '2016-01-01-2']), traffic_dict=traffic_dict,pre_num = 3).tolist()
        assert [2246,2081] == self.find_prev_traffic(pd.Series([1, '2016-01-01-9']), traffic_dict=traffic_dict,pre_num = 2).tolist()
        
        data_dir = g_singletonDataFilePath.getTest1Dir()
        traffic_dict = self.get_traffic_dict(data_dir)
        assert [346,424,0] == self.find_prev_traffic(pd.Series([66, '2016-01-30-141']), traffic_dict=traffic_dict,pre_num = 3).tolist()
        assert [501,484,447] == self.find_prev_traffic(pd.Series([66, '2016-01-30-70']), traffic_dict=traffic_dict,pre_num = 3).tolist()
        assert [772,802,775] == self.find_prev_traffic(pd.Series([57, '2016-01-24-58']), traffic_dict=traffic_dict,pre_num = 3).tolist()
        
        
        print 'passed unit test'
        
        
        return
    def get_intial_colnames(self):
        return ['start_district_hash', 'level_1', 'level_2', 'level_3', 'level_4','Time']
    def get_traffic_dict(self,data_dir):
        t0 = time()
        filename = '../data_raw/' + data_dir.split('/')[-2]  + '_traffic.csv.dict.pickle'
        dumpload = DumpLoad( filename)
        if dumpload.isExisiting():
            return dumpload.load()
        
        resDict = {}
        df = self.load_trafficdf(data_dir)
        for _, row in df.iterrows():
            resDict[tuple(row[['start_district_id','time_slotid']].tolist())] = row['traffic']
        
        dumpload.dump(resDict)
        print "dump traffic dict:", round(time()-t0, 3), "s"
        return resDict
    def process_all_df(self, df):
        self.add_timeid_col(df)
        self.add_timedate_col(df)
        self.sort_by_district_time(df)
        df = df.drop('start_district_hash', axis=1, inplace = True)
        return
    def load_trafficdf(self, dataDir):
        filename = dataDir + 'traffic_data/temp/traffic.csv'
        return pd.read_csv(filename, index_col= 0)

    def cal_traffic_value(self, level_series):
        res = 0
        for _, item in level_series.iteritems():
            item = item.split(':')
            res = res + int(item[0]) * int(item[1])            
        return res
    def find_prev_traffic(self, series, traffic_dict=None,pre_num = 2):
        start_district_id = series.iloc[0]
        time_slotid = series.iloc[1]
        index = ['traffic' + str(i + 1) for i in range(pre_num)]
        res = []
        prevSlots = singletonTimeslot.getPrevSlots(time_slotid, pre_num)
        for prevslot in prevSlots:
            try:
                res.append(traffic_dict[(start_district_id, prevslot)])
            except:
                res.append(0)
        return pd.Series(res, index = index)

    
    def process_one_df(self, df):
        df['start_district_id'] = df['start_district_hash'].map(singletonDistricId.convertToId)
        df['traffic'] = df[['level_1', 'level_2', 'level_3']].apply(self.cal_traffic_value, axis = 1,)
        #Remove duplicate time_slotid, retain the ealier ones
#         df.drop_duplicates(subset='time_slotid', keep='first', inplace=True)
        return
    


if __name__ == "__main__":   
    obj= ExploreTraffic()
    obj.run()