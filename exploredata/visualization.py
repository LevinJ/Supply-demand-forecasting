from order import ExploreOrder
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(color_codes=True)
from utility.datafilepath import g_singletonDataFilePath
from weather import ExploreWeather
import numpy as np
import math


class visualizeOrder(ExploreOrder, ExploreWeather):
    def __init__(self):
        ExploreOrder.__init__(self)
#         self.gapdf, _ = self.loadGapData(g_singletonDataFilePath.getGapCsv_Train())
        self.gapdf = self.loadGapCsvFile(g_singletonDataFilePath.getGapCsv_Train())
        self.gap_time_dict = self.gapdf.groupby('time_slotid')['gap'].sum().to_dict()
        self.weathdf = self.load_weatherdf(g_singletonDataFilePath.getTrainDir())
#         self.gapDict = self.loadGapDict(g_singletonDataFilePath.getTrainDir() + 'temp/gap.csv.dict.pickle')
        return
    def drawGapDistribution(self):
        self.gapdf[self.gapdf['gapdf'] < 10]['gapdf'].hist(bins=50)
#         sns.distplot(self.gapdf['gapdf']);
#         sns.distplot(self.gapdf['gapdf'], hist=True, kde=False, rug=False)
#         plt.hist(self.gapdf['gapdf'])
        plt.show()
        return
    def drawGapCorrelation(self):
        _, (ax1, ax2) = plt.subplots(nrows=2, ncols=1)
        res = self.gapdf.groupby('start_district_id')['gapdf'].sum()
        ax1.bar(res.index, res.values)
        res = self.gapdf.groupby('time_slotid')['gapdf'].sum()
        ax2.bar(res.index.map(lambda x: x[11:]), res.values)
        plt.show()
        return
    def find_gap_by_timeslot(self, timeslot):
        try:
            return self.gap_time_dict[timeslot]
        except:
            return 0
        return
    def show_weather_bydate(self):
        self.weathdf['gap'] = self.weathdf['time_slotid'].apply(self.find_gap_by_timeslot)
        by_date = self.weathdf.groupby('time_date')
        size = len(by_date)
        col_len = row_len = math.ceil(math.sqrt(size))
        count = 1
        for name, group in by_date:
            ax=plt.subplot(row_len, col_len, count)
#             temp = np.empty(group['time_id'].shape[0])
#             temp.fill(2)
            
#             ax.plot(group['time_id'], group['gap']/group['gap'].max(), 'r', alpha=0.75)
#             ax.plot(group['time_id'], group['weather']/group['weather'].max())
            ax.bar(group['time_id'], group['weather'], width=1)
            ax.set_title(name)
            count = count + 1
#             plt.bar(group['time_id'], np.full(group['time_id'].shape[0], 5), width=1)
            
        plt.show()
        return
    def run(self):
        self.show_weather_bydate()
#         self.drawGapDistribution()
#         self.drawGapCorrelation()
        return
    


if __name__ == "__main__":   
    obj= visualizeOrder()
    obj.run()