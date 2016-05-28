import pandas as pd
from  districtid import singletonDistricId
from timeslot import singletonTimeslot




class ExploreOrder:
    def __init__(self):
        return
    
    
    def loadorder(self, filename):
        df = pd.read_csv(filename, delimiter='\t', header=None, names =['order_id','driver_id','passenger_id', 'start_district_hash','dest_district_hash','Price','Time'])
        df['start_district_id'] = df['start_district_hash'].map(singletonDistricId.convertToId)
        df['time_slotid'] = df['Time'].map(singletonTimeslot.convertToSlot)
#         df['dest_district_id'] = df['dest_district_hash'].map(singletonDistricId.convertToId)
        df.to_csv(filename + '.csv')
        print df.describe()
        return
    def run(self):
        self.loadorder('../data/citydata/season_1/training_data/order_data/order_data_2016-01-03')
        return

 

if __name__ == "__main__":   
    obj= ExploreOrder()
    obj.run()
#     obj = districtid.singletonObj
#     print obj.convertToId('a5609739c6b5c2719a3752327c5e33a7')