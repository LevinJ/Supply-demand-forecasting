from order import ExploreOrder
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(color_codes=True)
from utility.datafilepath import g_singletonDataFilePath



class visualizeOrder(ExploreOrder):
    def __init__(self):
        ExploreOrder.__init__(self)
#         gapCsvFilePath = g_singletonDataFilePath.getGapCsv_Train()
        gapCsvFilePath = g_singletonDataFilePath.getGapCsv_Test1()
        self.df = pd.read_csv(gapCsvFilePath)
        print self.df.describe()
        return
    def drawOrderDistribution(self):
#         self.df['gap'].hist(bins=200)
#         sns.distplot(self.df['gap']);
        sns.distplot(self.df['gap'], hist=False, kde=True, rug=False)
#         plt.hist(self.df['gap'])
        plt.show()
        return
    def drawOderCorrelation(self):
        _, (ax1, ax2) = plt.subplots(nrows=2, ncols=1)
        res = self.df.groupby('start_district_id')['gap'].sum()
        ax1.bar(res.index, res.values)
        res = self.df.groupby('time_slotid')['gap'].sum()
        ax2.bar(res.index.map(lambda x: x[11:]), res.values)
        plt.show()
        return
    def run(self):
#         self.drawOrderDistribution()
        self.drawOderCorrelation()
        return
    


if __name__ == "__main__":   
    obj= visualizeOrder()
    obj.run()