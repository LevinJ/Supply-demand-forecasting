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
#         self.df, _ = self.loadGapData(g_singletonDataFilePath.getGapCsv_Train())
#         self.df, _ = self.loadGapData(g_singletonDataFilePath.getGapCsv_Test1())
        self.weathdf = self.load_weatherdf(g_singletonDataFilePath.getTrainDir())
        return
    def drawGapDistribution(self):
        self.df[self.df['gap'] < 10]['gap'].hist(bins=50)
#         sns.distplot(self.df['gap']);
#         sns.distplot(self.df['gap'], hist=True, kde=False, rug=False)
#         plt.hist(self.df['gap'])
        plt.show()
        return
    def drawGapCorrelation(self):
        _, (ax1, ax2) = plt.subplots(nrows=2, ncols=1)
        res = self.df.groupby('start_district_id')['gap'].sum()
        ax1.bar(res.index, res.values)
        res = self.df.groupby('time_slotid')['gap'].sum()
        ax2.bar(res.index.map(lambda x: x[11:]), res.values)
        plt.show()
        return
    def show_weather_bydate(self):
        by_date = self.weathdf.groupby('time_date')
        size = len(by_date)
        col_len = row_len = math.ceil(math.sqrt(size))
        count = 1
        for name, group in by_date:
            ax=plt.subplot(row_len, col_len, count)
            temp = np.empty(group['time_id'].shape[0])
            temp.fill(2)
            ax.bar(group['time_id'], temp, width=1)
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