from order import ExploreOrder
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(color_codes=True)
from utility.datafilepath import g_singletonDataFilePath
from visualization import visualizeData
import numpy as np
import math
import pandas as pd
from utility.dumpload import DumpLoad



class VisualizeTrainData(visualizeData):
    def __init__(self):
        ExploreOrder.__init__(self)
        visualizeData.__init__(self)
       
        
#         self.weathdf = self.load_weatherdf(data_dir)
        
        return
    def univariate(self):
#         self.gapdistricution()
#         self.weather_distribution()
#         self.traffic_districution()
        self.poi_distribution()
        
        return
    def poi_distribution(self):
#         dt_list = self.get_district_type_list()
#         dt_sum = pd.Series(np.zeros(len(dt_list)), index=dt_list)
#         dt_dict = self.get_district_type_dict()
#         for _, item_series in dt_dict.iteritems():
#             dt_sum = dt_sum + item_series
        df = self.get_district_type_table()
        dt_sum = df[self.get_district_type_list()].sum(axis = 0)
        print dt_sum.describe()
        dt_sum.plot(kind='bar')
        plt.xlabel('District Attribute')
        plt.title('Barplot of District Attribute')
        return
    def weather_distribution(self):
        data_dir = g_singletonDataFilePath.getTrainDir()
        self.gapdf = self.load_weatherdf(data_dir)
        print self.gapdf['weather'].describe()
#         sns.distplot(self.gapdf['gap'],kde=False, bins=100);
        
        sns.countplot(x="weather", data=self.gapdf, palette="Greens_d");
        plt.title('Countplot of Weather')
#         self.gapdf['weather'].plot(kind='bar')
#         plt.xlabel('Weather')
#         plt.title('Histogram of Weather')
        return
    
    def gapdistricution(self):
        data_dir = g_singletonDataFilePath.getTrainDir()
        self.gapdf = self.load_gapdf(data_dir)
        print self.gapdf['gap'].describe()
#         sns.distplot(self.gapdf['gap'],kde=False, bins=100);
        self.gapdf['gap'].plot(kind='hist', bins=200)
        plt.xlabel('Gaps')
        plt.title('Histogram of Gaps')

        return
    def traffic_districution(self):
        data_dir = g_singletonDataFilePath.getTrainDir()
        df = self.load_trafficdf(data_dir)
        print df['traffic'].describe()
#         sns.distplot(self.gapdf['gap'],kde=False, bins=100);
        df['traffic'].plot(kind='hist', bins=100)
        plt.xlabel('Traffic')
        plt.title('Histogram of Traffic')

        return
#     def disp_gap_bydistrict(self, disp_ids = np.arange(34,67,1), cls1 = 'start_district_id', cls2 = 'time_id'):
# #         disp_ids = np.arange(1,34,1)
#         plt.figure()
#         by_district = self.gapdf.groupby(cls1)
#         size = len(disp_ids)
# #         size = len(by_district)
#         col_len = row_len = math.ceil(math.sqrt(size))
#         count = 1
#         for name, group in by_district:
#             if not name in disp_ids:
#                 continue
#             plt.subplot(row_len, col_len, count)
#             group.groupby(cls2)['gap'].mean().plot()
#             count += 1   
#         return
    def disp_gap_byweather(self):
        df = self.gapdf
        data_dir = g_singletonDataFilePath.getTrainDir()
        dumpfile_path = '../data_preprocessed/' + data_dir.split('/')[-2] + '_prevweather.df.pickle'
        dumpload = DumpLoad(dumpfile_path)
        if dumpload.isExisiting():
            temp_df = dumpload.load()
        else:
            weather_dict = self.get_weather_dict(data_dir)
            
            temp_df = self.X_y_Df['time_slotid'].apply(self.find_prev_weather_mode, weather_dict=weather_dict)     
            dumpload.dump(temp_df)
            
        df = pd.concat([df, temp_df],  axis=1)
        
        gaps_mean = df.groupby('preweather')['gap'].mean()
        gaps_mean.plot(kind='bar')
        plt.ylabel('Mean of gap')
        plt.xlabel('Weather')
        plt.title('Weather/Gap Correlation')
        return
    
    def disp_gap_by_district_type(self):
        df = self.gapdf
        data_dir = g_singletonDataFilePath.getTrainDir()
        
        dumpfile_path = '../data_preprocessed/' + data_dir.split('/')[-2] + '_poi.df.pickle'
        dumpload = DumpLoad(dumpfile_path)
        if dumpload.isExisiting():
            temp_df = dumpload.load()
        else:
            poi_dict = self.get_district_type_dict()
            
            temp_df = self.X_y_Df[['start_district_id']].apply(self.find_poi,axis = 1, poi_dict=poi_dict)
            
            dumpload.dump(temp_df)
        df = pd.concat([df, temp_df],  axis=1)
        dt_list = self.get_district_type_list()
        
        size = len(dt_list)
        col_len = 4
        row_len = 7
#         col_len = row_len = int(math.ceil(math.sqrt(size)))
#         count = 1
        _, axarr = plt.subplots(row_len, col_len, sharex=True, sharey=True)
        for row in range(row_len):
            for col in range(col_len):
                index = row * col_len + col
                if index >= size:
                    break
                item = dt_list[index]
                axarr[row, col].scatter(df[item], df['gap'])
                axarr[row, col].set_ylabel('Gap')
                axarr[row, col].set_xlabel(item)
#                 axarr[row, col].set_title('POI/Gap Correlation')
#         for item in dt_list:
#             plt.subplot(row_len, col_len, count)
#             plt.scatter(df[item], df['gap'])
#             plt.ylabel('Gap')
#             plt.xlabel('POI')
#             count += 1
        
       
#         plt.title('POI/Gap Correlation')
        return
    def disp_district_by_district_type(self):
        df = self.get_district_type_table()
        dt_list = self.get_district_type_list()
        size = df.shape[0]
        col_len = 8
        row_len = 8
        
        _, axarr = plt.subplots(row_len, col_len, sharex=True, sharey=True)
        for row in range(row_len):
            for col in range(col_len):
                index = row * col_len + col
                if index >= size:
                    break
                item = df.iloc[index]
                x_locations = np.arange(len(dt_list))
                axarr[row, col].bar(x_locations, item[dt_list])
                axarr[row, col].set_xlabel('start_district_' + str(item['start_district_id']))
        return
    
    def disp_gap_bytraffic(self):
        df = self.gapdf
        data_dir = g_singletonDataFilePath.getTrainDir()
        dumpfile_path = '../data_preprocessed/' + data_dir.split('/')[-2] + '_prevtraffic.df.pickle'
        dumpload = DumpLoad(dumpfile_path)
        if dumpload.isExisiting():
            temp_df = dumpload.load()
        else:
            traffic_dict = self.get_traffic_dict(data_dir)
            
            temp_df = self.X_y_Df[['start_district_id', 'time_slotid']].apply(self.find_prev_traffic,axis = 1, traffic_dict=traffic_dict, pre_num = 3)   
            dumpload.dump(temp_df)
            
        df = pd.concat([df, temp_df],  axis=1)
     
        
        by_traffic = df.groupby('traffic1')
        x=[]
        y=[]
        for name, group in by_traffic:
            x.append(name)
            y.append(group['gap'].mean())
        plt.scatter(x,y)
        
        return
    def bivariate(self):
        self.disp_district_by_district_type()
#         self.disp_gap_by_district_type()
#         self.disp_gap_bytraffic()
#         self.disp_gap_byweather()
#         self.disp_gap_bydate()
#         self.disp_gap_bytimeiid()
#         self.disp_gap_bydistrict()
        return
    def run(self):
        self.bivariate()
#         self.univariate()
#         self.disp_by_weather()
#         self.disp_gap_bydistrict(cls2 = 'time_id')
#         self.disp_gap_bydistrict(cls2 = 'time_date')
        plt.show()
#         self.disp_gap_bytimeiid()
#         self.disp_gap_bydistrict()
#         self.disp_gap_bydate()
        return
    


if __name__ == "__main__":   
    obj= VisualizeTrainData()
    obj.run()