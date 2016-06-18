from order import ExploreOrder
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(color_codes=True)
from utility.datafilepath import g_singletonDataFilePath
from visualization import visualizeData
import numpy as np
import math
import pandas as pd


class VisualizeTrainData(visualizeData):
    def __init__(self):
        ExploreOrder.__init__(self)
        data_dir = g_singletonDataFilePath.getTrainDir()
        self.gapdf = self.load_gapdf(data_dir)
        self.weathdf = self.load_weatherdf(data_dir)
        self.gapdf.describe()
        return
    def disp_gap_bydistrict(self, disp_ids = np.arange(34,67,1), cls1 = 'start_district_id', cls2 = 'time_id'):
#         disp_ids = np.arange(1,34,1)
        plt.figure()
        by_district = self.gapdf.groupby(cls1)
        size = len(disp_ids)
#         size = len(by_district)
        col_len = row_len = math.ceil(math.sqrt(size))
        count = 1
        for name, group in by_district:
            if not name in disp_ids:
                continue
            plt.subplot(row_len, col_len, count)
            group.groupby(cls2)['gap'].mean().plot()
            count += 1   
        return
    def disp_by_weather(self):
        by_weatehr = self.weathdf.groupby('weather')
        res = []
        
        for name, group in by_weatehr:
            gaps = self.gapdf[self.gapdf['time_slotid'].isin(group['time_slotid'])].shape[0]
#             gaps = self.gapdf[self.gapdf['time_slotid'].isin(group['time_slotid'])]['gap'].mean()
            res.append([name, gaps])
        
        df = pd.DataFrame(res, columns = ['weather', 'gaps'])
#         plt.scatter(df['weather'], df['gaps'])
        plt.scatter(df['weather'], df['gaps'])
        return
    def run(self):
        self.disp_by_weather()
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