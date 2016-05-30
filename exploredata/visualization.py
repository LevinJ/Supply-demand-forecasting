from order import ExploreOrder
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(color_codes=True)
import os.path



class visualizeOrder(ExploreOrder):
    def __init__(self):
        ExploreOrder.__init__(self)
        self.orderFileDir = '../data/citydata/season_1/test_set_1/order_data/'
#         self.orderFileDir = '../data/citydata/season_1/training_data/order_data/'
        allOderFilePath = self.orderFileDir + 'temp/allorders.csv'
        if not os.path.exists(allOderFilePath):
            self.loadAllOrders()
            self.combineAllOrders()
        self.df = pd.read_csv(allOderFilePath)
        print self.df.describe()
        return
    def drawOrderDistribution(self):
#         self.df['missed_request'].hist(bins=200)
#         sns.distplot(self.df['missed_request']);
        sns.distplot(self.df['missed_request'], hist=False, kde=True, rug=False)
#         plt.hist(self.df['missed_request'])
        plt.show()
        return
    def drawOderCorrelation(self):
        _, (ax1, ax2) = plt.subplots(nrows=2, ncols=1)
        res = self.df.groupby('start_district_id')['missed_request'].sum()
        ax1.bar(res.index, res.values)
        res = self.df.groupby('time_slotid')['missed_request'].sum()
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