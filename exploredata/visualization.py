from order import ExploreOrder
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(color_codes=True)




class visualizeOrder(ExploreOrder):
    def __init__(self):
        ExploreOrder.__init__(self)
        self.df = pd.read_csv('../data/citydata/season_1/training_data/order_data/order_data_2016-01-03_gap.csv')
        print self.df.describe()
        return
    def drawOrderDistribution(self):
#         self.df['missed_request'].hist(bins=200)
#         sns.distplot(self.df['missed_request']);
        sns.distplot(self.df['missed_request'], hist=False, kde=True, rug=False)
#         plt.hist(self.df['missed_request'])
        plt.show()
        return
    def run(self):
        self.drawOrderDistribution()
        return
    


if __name__ == "__main__":   
    obj= visualizeOrder()
    obj.run()