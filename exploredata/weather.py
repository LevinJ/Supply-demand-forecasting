from utility.datafilepath import g_singletonDataFilePath
from timeslot import singletonTimeslot
import pandas as pd
import os
from exploredata import ExploreData
from time import time
from utility.dumpload import DumpLoad
from scipy.stats import mode
class ExploreWeather(ExploreData ):
    def __init__(self):
        return
    def run(self):
        self.__unittest()
        data_dir = g_singletonDataFilePath.getTest2Dir()
#         self.save_all_csv( data_dir+ 'weather_data/')
#         self.combine_all_csv(data_dir + 'weather_data/temp/', 'weather_', 'weather.csv')
        self.get_weather_dict(data_dir)
        return
    def __unittest(self):
        #         self.combine_all_csv(g_singletonDataFilePath.getTrainDir() + 'weather_data/temp/', 'weather_', 'weather.csv')
#         self.save_one_csv(g_singletonDataFilePath.getTrainDir() + 'weather_data/weather_data_2016-01-02')
#         weatherdf = self.load_weatherdf(g_singletonDataFilePath.getTrainDir())
        weather_dict = self.get_weather_dict(g_singletonDataFilePath.getTrainDir())
        assert  2== self.find_prev_weather_mode('2016-01-01-1', weather_dict = weather_dict)[0]
        assert  2== self.find_prev_weather_mode('2016-01-21-144', weather_dict = weather_dict)[0]
#         
        assert  2== self.find_prev_weather_mode('2016-01-21-115', weather_dict = weather_dict)[0]
        assert  2== self.find_prev_weather_mode('2016-01-21-114', weather_dict = weather_dict)[0]
        print 'passed unit test'
        
        
        return
    def get_weather_dict(self,data_dir):
        t0 = time()
        filename = '../data_raw/' + data_dir.split('/')[-2] + '_weather.csv.dict.pickle'
        dumpload = DumpLoad( filename)
        if dumpload.isExisiting():
            return dumpload.load()
        
        resDict = {}
        df = self.load_weatherdf(data_dir)
        for index, row in df.iterrows():
            resDict[row['time_slotid']] = (index, row['weather'], row['temparature'], row['pm25'])
        for name, group in df.groupby('time_date'):
            resDict[name] = (-1, mode(group['weather'])[0][0], mode(group['temparature'])[0][0], mode(group['pm25'])[0][0])
            
       
        dumpload.dump(resDict)
        print "dump weather dict:", round(time()-t0, 3), "s"
        return resDict
    def process_all_df(self, df):
        self.add_timeid_col(df)
        self.add_timedate_col(df)
        self.sort_by_time(df)
        return
    def get_intial_colnames(self):
        return ['Time','weather','temparature', 'pm25']
    def load_weatherdf(self, dataDir):
        filename = dataDir + 'weather_data/temp/weather.csv'
        return pd.read_csv(filename, index_col= 0)
    def is_first_record(self, weather_dict, time_slotid):
        try:
            res = weather_dict[time_slotid]
            if (res[0] == 0):
                return True
        except:
            pass
        return False
    def find_prev_weather(self, time_slotid, weather_dict=None,):
        if self.is_first_record(weather_dict, time_slotid):
            return pd.Series([0], index = ['preweather'])
        current_slot = time_slotid
        while(True):
            res = singletonTimeslot.getPrevSlots(current_slot, 1)
            current_slot = res[0]
            try:
                res = weather_dict[current_slot]
                return  pd.Series([res[1]], index = ['preweather'])
            except:
                pass
        return
    def find_prev_weather_mode(self, time_slotid, weather_dict=None,):
        try:
            prev_slot = singletonTimeslot.getPrevSlots(time_slotid, 1)[0]
            res = weather_dict[prev_slot]
        except:
            current_date = singletonTimeslot.getDate(time_slotid)
            res = weather_dict[current_date]
        return pd.Series([res[1]], index = ['preweather'])
    def process_one_df(self, df):
        #Remove duplicate time_slotid, retain the ealier ones
        df.drop_duplicates(subset='time_slotid', keep='first', inplace=True)
        return
    


if __name__ == "__main__":   
    obj= ExploreWeather()
    obj.run()