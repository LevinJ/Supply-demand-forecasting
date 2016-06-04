import pandas as pd
from  districtid import singletonDistricId
from timeslot import singletonTimeslot
from os import walk
import os.path
from utility.datafilepath import g_singletonDataFilePath
from time import time
from utility.dumpload import DumpLoad




class ExploreOrder:
    def __init__(self):
        return
    
    def saveAllGapCsv(self, orderFileDir):
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
        filePaths = self.getAllFilePaths(orderFileDir + 'temp/')
        for filename in filePaths:
            if not filename.endswith('_gap.csv'):
                continue
            df = pd.read_csv(filename, index_col=0)
            resDf = pd.concat([resDf, df], ignore_index = True)
        resDf = self.sortGapRows(resDf)
        self.addTimeIdColumn(resDf)
        resDf.to_csv(orderFileDir + 'temp/'+ 'gap.csv')
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
    def loadGapData(self, gapFileName):
        """
        This is the only interface taht should be called by outsider 
        It returns the raw csv file and index hash of district/timeslot for quick retrieval of previous gaps
        """
        return (self.loadGapCsvFile(gapFileName), self.loadGapDict(gapFileName))
    def loadGapCsvFile(self, fileName):
        if not os.path.exists(fileName):
            orderFileDir = os.path.dirname(os.path.dirname(fileName)) + r"/"
            self.saveAllGapCsv(orderFileDir)
            self.combineAllGapCsv(orderFileDir)
        df = pd.read_csv(fileName, index_col= 0)
        print df.describe()
        return df
    def loadGapDict(self, fileName):
        t0 = time()
        dumpload = DumpLoad( fileName)
        if dumpload.isExisiting():
            return dumpload.load()
        
        resDict = {}
        df = self.loadGapCsvFile(fileName)
        for _, row in df.iterrows():
            resDict[tuple(row[['start_district_id','time_slotid']].tolist())] = row['gap']
        
        dumpload.dump(resDict)
        print "dump gapdict:", round(time()-t0, 3), "s"
        return resDict
    def dispInfoAboutGap(self):
        df = self.loadGapCsvFile(g_singletonDataFilePath.getGapCsv_Train())
        print "Number of Gaps with zero value {}, {}".format((df['gap'] == 0).sum(), (df['gap'] == 0).sum()/float(df.shape[0]))
        return
    def run(self):
#         res = self.loadGapData(g_singletonDataFilePath.getGapCsv_Train())
        res = self.loadGapData(g_singletonDataFilePath.getGapCsv_Test1())
        
        return

 

if __name__ == "__main__":   
    obj= ExploreOrder()
    obj.run()
#     obj = districtid.singletonObj
#     print obj.convertToId('a5609739c6b5c2719a3752327c5e33a7')