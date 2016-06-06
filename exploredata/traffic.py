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
        self.__unittest()
#         self.save_all_csv(g_singletonDataFilePath.getTrainDir()+ 'traffic_data/')
#         self.combine_all_csv(g_singletonDataFilePath.getTrainDir() + 'traffic_data/temp/', 'traffic_', 'traffic.csv')
        self.get_traffic_dict(g_singletonDataFilePath.getTrainDir())
        return
    def __unittest(self):
        #         self.combine_all_csv(g_singletonDataFilePath.getTrainDir() + 'weather_data/temp/', 'weather_', 'weather.csv')
#         self.save_one_csv(g_singletonDataFilePath.getTrainDir() + 'traffic_data/traffic_data_2016-01-04')
#         weatherdf = self.load_weatherdf(g_singletonDataFilePath.getTrainDir())
#         weather_dict = self.get_weather_dict(g_singletonDataFilePath.getTrainDir())
#         assert  0== self.find_prev_weather('2016-01-01-1', weather_dict = weather_dict)
#         assert  2== self.find_prev_weather('2016-01-21-144', weather_dict = weather_dict)
#         
#         assert  4== self.find_prev_weather('2016-01-21-115', weather_dict = weather_dict)
#         assert  4== self.find_prev_weather('2016-01-21-114', weather_dict = weather_dict)
        print 'passed unit test'
        
        
        return
    def get_intial_colnames(self):
        return ['start_district_hash', 'level_1', 'level_2', 'level_3', 'level_4','Time']
    def get_traffic_dict(self,dataDir):
        t0 = time()
        filename = dataDir + 'traffic_data/temp/traffic.csv.dict.pickle'
        dumpload = DumpLoad( filename)
        if dumpload.isExisiting():
            return dumpload.load()
        
        resDict = {}
        df = self.load_trafficdf(dataDir)
        for _, row in df.iterrows():
            resDict[tuple(row[['start_district_id','time_slotid']].tolist())] = row['traffic']
        
        dumpload.dump(resDict)
        print "dump traffic dict:", round(time()-t0, 3), "s"
        return resDict
    def process_all_df(self, df):
#         self.add_timeid_col(df)
#         self.add_timedate_col(df)
        self.sort_by_district_time(df)
        df = df.drop('start_district_hash', axis=1, inplace = True)
        return
    def load_trafficdf(self, dataDir):
        filename = dataDir + 'traffic_data/temp/traffic.csv'
        return pd.read_csv(filename, index_col= 0)
    def is_first_record(self, weather_dict, time_slotid):
        try:
            res = weather_dict[time_slotid]
            if (res[0] == 0):
                return True
        except:
            pass
        return False
    def cal_traffic_value(self, level_series):
        res = 0
        for _, item in level_series.iteritems():
            item = item.split(':')
            res = res + int(item[0]) * int(item[1])            
        return res
    def find_prev_weather(self, time_slotid, weather_dict=None,):
        if self.is_first_record(weather_dict, time_slotid):
            return 0
        current_slot = time_slotid
        while(True):
            res = singletonTimeslot.getPrevSlots(current_slot, 1)
            current_slot = res[0]
            try:
                res = weather_dict[current_slot]
                return res[1]
            except:
                pass
        return
    
    def process_one_df(self, df):
        df['start_district_id'] = df['start_district_hash'].map(singletonDistricId.convertToId)
        df['traffic'] = df[['level_1', 'level_2', 'level_3']].apply(self.cal_traffic_value, axis = 1,)
        #Remove duplicate time_slotid, retain the ealier ones
#         df.drop_duplicates(subset='time_slotid', keep='first', inplace=True)
        return
    


if __name__ == "__main__":   
    obj= ExploreTraffic()
    obj.run()