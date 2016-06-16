from order import ExploreOrder
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(color_codes=True)
from utility.datafilepath import g_singletonDataFilePath
from visualization import visualizeData
import numpy as np
import math


class VisualizeBothData(visualizeData):
    def __init__(self):
        ExploreOrder.__init__(self)
        self.gap_testdf = self.load_gapdf(g_singletonDataFilePath.getTest1Dir())
        self.gap_traindf = self.load_gapdf(g_singletonDataFilePath.getTrainDir())
        self.gap_traindf.describe()
        self.gap_testdf.describe()
        return
    def disp_bydate(self):
        plt.figure(1)
        plt.subplot(2,1,1)
        self.gap_traindf.groupby('time_date')['gap'].mean().plot(kind='bar')
        plt.subplot(2,1,2)
        self.gap_testdf.groupby('time_date')['gap'].mean().plot(kind='bar')
        plt.show()
        return
    def disp_bydistrictid(self):
        
        _, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
        gaps_mean = self.gap_traindf.groupby('start_district_id')['gap'].sum()
        for i in gaps_mean.index:
            ax1.plot([i,i], [0, gaps_mean[i]], 'k-')
        
        
        gaps_mean = self.gap_testdf.groupby('start_district_id')['gap'].sum()
        for i in gaps_mean.index:
            ax2.plot([i,i], [0, gaps_mean[i]], 'k-')
            
            
        plt.show()
        return
    def disp_timeid(self):
        
        _, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
        gaps_mean = self.gap_traindf.groupby('time_id')['gap'].mean()
        for i in gaps_mean.index:
            ax1.plot([i,i], [0, gaps_mean[i]], 'k-')
        
        
        gaps_mean = self.gap_testdf.groupby('time_id')['gap'].mean()
        for i in gaps_mean.index:
            ax2.plot([i,i], [0, gaps_mean[i]], 'k-')
            
            
        plt.show()
        return
    def disp_gap_series(self, storeid):
        return
    def disp_gap_by_dateandslot(self, df):
        plt.figure()
        by_date = df.groupby('time_date')
        size = len(by_date)
        col_len = row_len = math.ceil(math.sqrt(size))
        count = 1
        for name, group in by_date:
            ax = plt.subplot(row_len, col_len, count)
            gap_mean = group.groupby('time_id')['gap'].mean()
            ax.scatter(gap_mean.index, gap_mean.values)
            ax.set_title(name)
            count = count + 1
             
#         plt.show()
        return
    def run(self):
#         self.disp_gap_by_dateandslot(self.gap_traindf)
#         self.disp_gap_by_dateandslot(self.gap_testdf)
        
#         self.disp_gap_series(51)
#         self.disp_timeid()
#         self.disp_bydate()
        self.disp_bydistrictid()
        plt.show()
#         self.disp_gap_bytimeiid()
#         self.disp_gap_bydistrict()
#         self.disp_gap_bydate()
        return
    


if __name__ == "__main__":   
    obj= VisualizeBothData()
    obj.run()