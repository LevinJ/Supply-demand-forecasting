from exploredata.order import ExploreOrder
from utility.datafilepath import g_singletonDataFilePath
from time import time
from utility.dumpload import DumpLoad
import pandas as pd
from scipy.stats import mode
import numpy as np

class HistoricalData(object):
    def __init__(self):
        
        return
    def run(self):
        res = self.get_history_data_dict()
        print len(res)
        return
    def get_history_data_dict(self):
        """
        indexes for quick search
        key = 'start_district_id','time_id'
        value = 'gap
        its data includes those from train, test1, test2.
        """
        t0 = time()

        filename = "../data_preprocessed/"  + 'traintest_history_data.dict.pickle'
        
        dumpload = DumpLoad( filename)
        if dumpload.isExisiting():
            return dumpload.load()
        
        test1data_df = ExploreOrder().load_gapdf(g_singletonDataFilePath.getTest1Dir())
        test2data_df = ExploreOrder().load_gapdf(g_singletonDataFilePath.getTest2Dir())
        traindata_df = ExploreOrder().load_gapdf(g_singletonDataFilePath.getTrainDir())
        
        
        df = pd.concat([traindata_df, test1data_df,test2data_df],  axis=0)
        self.__fileter_earlier_date(df)
        res_dict = self.__generate_dict(df)            
       
        dumpload.dump(res_dict)
        print "dump weather dict:", round(time()-t0, 3), "s"
        return  res_dict
    def __fileter_earlier_date(self, df):
        return
    def __get_historylist_from_dict(self,history_dict,start_district_id, time_id):
        res = [0]
        try:
            res = history_dict[(start_district_id, time_id)]
            if len(res) == 0:
                res = [0]
        except:
            pass
        return res
    def find_history_data(self, row, history_dict=None,):
        start_district_id = row.iloc[0]
        time_id = row.iloc[1]
        index = ['history_mean','history_median','history_mode','history_plus_mean','history_plus_median', 'history_plus_mode']

        min_list = self.__get_historylist_from_dict(history_dict, start_district_id, time_id)
        plus_list1 = self.__get_historylist_from_dict(history_dict, start_district_id, time_id-1)
        plus_list2 = self.__get_historylist_from_dict(history_dict, start_district_id, time_id-2)
        plus_list = np.array((plus_list1 + plus_list2 + min_list))
        min_list = np.array(min_list)
        
        res =pd.Series([min_list.mean(), np.median(min_list), mode(min_list)[0][0], plus_list.mean(), np.median(plus_list),mode(plus_list)[0][0]], index = index)
        
        return res
    
        return pd.Series(res, index = ['history_mean', 'history_mode', 'history_median'])
    def __generate_dict(self, df):
        res_dict ={}
        by_group = df.groupby(['start_district_id','time_id'])
        for name, row in by_group:
            res_dict[name] = row['gap'].tolist()
        return res_dict
    def get_bydistrict_time_dict(self):
        return
    
    
    def __unit_test(self):
        self.get_history_data_dict()
        return



if __name__ == "__main__":   
    obj= HistoricalData()
    obj.run()