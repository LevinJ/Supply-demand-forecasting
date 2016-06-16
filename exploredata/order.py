import pandas as pd
from  districtid import singletonDistricId
from timeslot import singletonTimeslot
from os import walk
import os.path
from utility.datafilepath import g_singletonDataFilePath
from time import time
from utility.dumpload import DumpLoad
import numpy as np




class ExploreOrder:
    def __init__(self):
        return
    
    def saveAllGapCsv(self, data_dir):
        orderFileDir = data_dir + 'order_data/'
        filePaths = self.getAllFilePaths(orderFileDir)
        for filename in filePaths:
            print "save gap csv for :{}".format(filename)
            res = self.saveOrderCsv(filename)
            self.saveGapCSV(*res)
        return
    def sortGapRows(self, df):
        df['timeslotrank'] = df['time_slotid'].map(lambda x: "-".join(x.split('-')[:3] + [x.split('-')[-1].zfill(3)]))
        df = df.sort_values(by = ['start_district_id','timeslotrank'])
        df = df.drop('timeslotrank', 1)
        df = df.reset_index(drop=True)
        return df
    def addTimeIdColumn(self, df):
        df['time_id'] = df['time_slotid'].apply(singletonTimeslot.getTimeId)
        return
    def combineAllGapCsv(self, orderFileDir):
        print "Combin all gaps"
        resDf = pd.DataFrame()
        filePaths = self.getAllFilePaths(orderFileDir + 'order_data/temp/')
        for filename in filePaths:
            if not filename.endswith('_gap.csv'):
                continue
            df = pd.read_csv(filename, index_col=0)
            resDf = pd.concat([resDf, df], ignore_index = True)
        resDf = self.sortGapRows(resDf)
        self.addTimeIdColumn(resDf)
        resDf['time_date'] = resDf['time_slotid'].apply(singletonTimeslot.getDate)
        resDf.to_csv(orderFileDir + 'order_data/temp/'+ 'gap.csv')
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
    def loadGapData(self, data_dir):
        """
        This is the only interface taht should be called by outsider 
        It returns the raw csv file and index hash of district/timeslot for quick retrieval of previous gaps
        """
        return (self.load_gapdf(data_dir), self.get_gap_dict(data_dir))
    def load_gapdf(self, data_dir):
        filename = data_dir + 'order_data/temp/gap.csv'
        df = pd.read_csv(filename, index_col= 0)
#         print df.describe()
        return df
    def get_gap_dict(self, data_dir):
        t0 = time()
        filename = data_dir + 'order_data/temp/gap.csv.dict.pickle'
        dumpload = DumpLoad( filename)
        if dumpload.isExisiting():
            return dumpload.load()
        
        resDict = {}
        df = self.load_gapdf(data_dir)
        for _, row in df.iterrows():
            resDict[tuple(row[['start_district_id','time_slotid']].tolist())] = row['gap']
        
        dumpload.dump(resDict)
        print "dump gapdict:", round(time()-t0, 3), "s"
        return resDict
    def dispInfoAboutGap(self):
        df = self.load_gapdf(g_singletonDataFilePath.getGapCsv_Train())
        print "Number of Gaps with zero value {}, {}".format((df['gap'] == 0).sum(), (df['gap'] == 0).sum()/float(df.shape[0]))
        return
    def unitTest(self):
        # test cases for find_prev_gap
        data_dir = g_singletonDataFilePath.getTrainDir()
        gap_dict = self.get_gap_dict(data_dir)
        assert [3096,1698,318,33,0,0] == self.find_prev_gap(pd.Series([51, '2016-01-01-5']), pre_num = 6, gap_dict = gap_dict).tolist()
        assert [0,0,0] == self.find_prev_gap(pd.Series([45, '2016-01-16-2']), pre_num = 3, gap_dict = gap_dict).tolist()
        assert [24,26,37] == self.find_prev_gap(pd.Series([53, '2016-01-04-56']), pre_num = 3, gap_dict = gap_dict).tolist()
        
        
        data_dir = g_singletonDataFilePath.getTest1Dir()
        gap_dict = self.get_gap_dict(data_dir)
        assert [0,1,0] == self.find_prev_gap(pd.Series([54, '2016-01-24-81']), pre_num = 3, gap_dict = gap_dict).tolist()
        assert [6,4,0] == self.find_prev_gap(pd.Series([7, '2016-01-30-141']), pre_num = 3, gap_dict = gap_dict).tolist()
        assert [0,0] == self.find_prev_gap(pd.Series([7, '2016-01-30-138']), pre_num = 2, gap_dict = gap_dict).tolist()
        assert [0,0,0] == self.find_prev_gap(pd.Series([7, '2016-01-30-139']), pre_num = 3, gap_dict = gap_dict).tolist()
        assert [0,0,1] == self.find_prev_gap(pd.Series([50, '2016-01-30-143']), pre_num = 3, gap_dict = gap_dict).tolist()
        assert [245,282,0] == self.find_prev_gap(pd.Series([51, '2016-01-22-141']), pre_num = 3, gap_dict = gap_dict).tolist()
        
        gap_meanmedian_dict = self.get_gap_meanmedian_dict()
        self.find_gap_meanmedian(pd.Series([5,55]),gap_meanmedian_dict = gap_meanmedian_dict)

        print "unit test passed"
        return
    def find_prev_gap(self, row, pre_num = 3, gap_dict = None):
        start_district_id = row.iloc[0]
        time_slotid = row.iloc[1]
        index = ['gap' + str(i + 1) for i in range(pre_num)]
        res = []
        prevSlots = singletonTimeslot.getPrevSlots(time_slotid, pre_num)
        for prevslot in prevSlots:
            try:
                res.append(gap_dict[(start_district_id, prevslot)])
            except:
                res.append(0)
        res =pd.Series(res, index = index)
        return res
    def get_gap_meanmedian_dict(self):
        data_dir = g_singletonDataFilePath.getTrainDir()
        filename = data_dir + 'order_data/temp/gap_meanmedian.dict.pickle'
        dumpload = DumpLoad( filename)
        if dumpload.isExisiting():
            return dumpload.load()
        
        resDict = {}
        df = self.load_gapdf(data_dir)
        grps = df.groupby(['start_district_id','time_id'])
        for name, row in grps:
            resDict[name] = row['gap'].tolist()
#             resDict[name] = [i for i in row['gap'].tolist() if i !=0]
            
        
        dumpload.dump(resDict)
        return resDict 
    def get_gapmeanmedainlist_from_dict(self,gap_meanmedian_dict,start_district_id, time_id):
        res = [0]
        try:
            res = gap_meanmedian_dict[(start_district_id, time_id)]
            if len(res) == 0:
                res = [0]
        except:
            pass
        return res
    def find_gap_meanmedian(self, row,gap_meanmedian_dict = None):
        start_district_id = row.iloc[0]
        time_id = row.iloc[1]
        index = ['mean','median','plus_mean','plus_median']

        min_list = self.get_gapmeanmedainlist_from_dict(gap_meanmedian_dict, start_district_id, time_id)
        plus_list1 = self.get_gapmeanmedainlist_from_dict(gap_meanmedian_dict, start_district_id, time_id-1)
        plus_list2 = self.get_gapmeanmedainlist_from_dict(gap_meanmedian_dict, start_district_id, time_id+1)
        plus_list = np.array((plus_list1 + plus_list2 + min_list))
        min_list = np.array(min_list)
        
        res =pd.Series([min_list.mean(), np.median(min_list), plus_list.mean(), np.median(plus_list)], index = index)
        
        return res
    def run(self):
        self.unitTest()
#         self.get_gap_meanmedian_dict()
#         data_dir = g_singletonDataFilePath.getTest2Dir()
#         self.saveAllGapCsv(data_dir)
#         self.combineAllGapCsv(data_dir)
#         res = self.get_gap_dict(data_dir)
        
        return

 

if __name__ == "__main__":   
    obj= ExploreOrder()
    obj.run()
#     obj = districtid.singletonObj
#     print obj.convertToId('a5609739c6b5c2719a3752327c5e33a7')