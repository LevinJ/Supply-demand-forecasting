from utility.datafilepath import g_singletonDataFilePath
from timeslot import singletonTimeslot
import pandas as pd
import os
from exploredata import ExploreData
class ExploreWeather(ExploreData ):
    def __init__(self):
        return
    def run(self):
        self.__unittest()
        return
    def __unittest(self):
        self.combine_all_csv(g_singletonDataFilePath.getTrainDir() + 'weather_data/temp/', 'weather_', 'weather.csv')
#         self.save_one_csv(g_singletonDataFilePath.getTrainDir() + 'weather_data/weather_data_2016-01-02')
#         self.save_all_csv(g_singletonDataFilePath.getTrainDir() + 'weather_data/')
        return
    def process_all_df(self, df):
        self.add_timeid_col(df)
        self.add_timedate_col(df)
        self.sort_by_time(df)
        return
    def load_weatherdf(self, dataDir):
        filename = dataDir + 'weather_data/temp/weather.csv'
        return pd.read_csv(filename, index_col= 0)
        return
    def process_one_df(self, df):
        #Remove duplicate time_slotid, retain the ealier ones
        df.drop_duplicates(subset='time_slotid', keep='first', inplace=True)
        return
    


if __name__ == "__main__":   
    obj= ExploreWeather()
    obj.run()