import pandas as pd
import os




class ExploreOrder:
    def __init__(self):
        return
    
    
    def loadorder(self, filename):
        df = pd.read_csv(filename, delimiter='\t', header=None, names =['order_id','driver_id','passenger_id', 'start_district_hash','dest_district_hash','Price','Time'])
        df.to_csv('order.csv')
        print df.describe()
        return
    def run(self):
        self.loadorder('../data/citydata/season_1/training_data/order_data/order_data_2016-01-03')
        return



if __name__ == "__main__":   
    obj= ExploreOrder()
    obj.run()