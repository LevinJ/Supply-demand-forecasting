import pandas as pd
from  districtid import singletonDistricId
from timeslot import singletonTimeslot
from os import walk
import os.path
from utility.datafilepath import g_singletonDataFilePath




class ExploreOrder:
    def __init__(self):
#         self.orderFileDir = g_singletonDataFilePath.getOrderDir_Train()# for train data
        self.orderFileDir = g_singletonDataFilePath.getOrderDir_Test1() # for test set1
        return
    
    def saveAllGapCsv(self):
        filePaths = self.getAllFilePaths(self.orderFileDir)
        for filename in filePaths:
            print "save gap csv for :{}".format(filename)
            res = self.saveOrderCsv(filename)
            self.saveGapCSV(*res)
        return
    def combineAllGapCsv(self):
        print "Combin all gaps"
        resDf = pd.DataFrame()
        filePaths = self.getAllFilePaths(self.orderFileDir + 'temp/')
        for filename in filePaths:
            if not filename.endswith('_gap.csv'):
                continue
            df = pd.read_csv(filename, index_col=0)
            resDf = pd.concat([resDf, df], ignore_index = True)
        resDf.to_csv(self.orderFileDir + 'temp/'+ 'gap.csv')
        print "Overall gap statistics: \n{}".format(resDf.describe())
        return
    def getAllFilePaths(self, rootpath):
        f = []
        for (dirpath, _, filenames) in walk(rootpath):
            #remove all files with file name starting with .
            filenames = [dirpath+x for x in filenames if x[0] != '.']
            f.extend(filenames)
            break
        return f
    def saveOrderCsv(self, filename):
        print("loading file {}".format(filename))
        df = pd.read_csv(filename, delimiter='\t', header=None, names =['order_id','driver_id','passenger_id', 'start_district_hash','dest_district_hash','Price','Time'])
        df['start_district_id'] = df['start_district_hash'].map(singletonDistricId.convertToId)
        df['time_slotid'] = df['Time'].map(singletonTimeslot.convertToSlot)
#         df['dest_district_id'] = df['dest_district_hash'].map(singletonDistricId.convertToId)
        df.to_csv(os.path.dirname(filename) + '/temp/'+ os.path.basename(filename) + '.csv')
        print df.describe()
        return df, filename
    def saveGapCSV(self, df, filename):
        items = []
        grouped  = df.groupby(['start_district_id','time_slotid'])
        for name, group in grouped:
            timeslot = singletonTimeslot.convertToStr(name[1])
            missedReqs = group['driver_id'].isnull().sum()
            allReqs = group.shape[0]
            items.append(list(name) + [timeslot]+[missedReqs] + [allReqs])
        resDf = pd.DataFrame(items, columns=['start_district_id','time_slotid','time_slot','gap', 'all_requests'])
        resDf.to_csv(os.path.dirname(filename) + '/temp/'+ os.path.basename(filename) + '_gap.csv')
        print resDf.describe()
        return
    def run(self):
        self.saveAllGapCsv()
        self.combineAllGapCsv()
        
#         res = self.saveOrderCsv('../data/citydata/season_1/training_data/order_data/order_data_2016-01-03')
#         self.saveGapCSV(*res)
        return

 

if __name__ == "__main__":   
    obj= ExploreOrder()
    obj.run()
#     obj = districtid.singletonObj
#     print obj.convertToId('a5609739c6b5c2719a3752327c5e33a7')