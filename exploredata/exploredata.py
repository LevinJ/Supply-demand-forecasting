from os import walk
from utility.datafilepath import g_singletonDataFilePath
from timeslot import singletonTimeslot
import pandas as pd
import os

class ExploreData(object):
    def get_all_file_paths(self, rootpath):
        f = []
        for (dirpath, _, filenames) in walk(rootpath):
            #remove all files with file name starting with .
            filenames = [dirpath+x for x in filenames if x[0] != '.']
            f.extend(filenames)
            break
        return f
    def combine_all_csv(self, dataDir, filename_prefix, filename_res):
        print "Combine all csv"
        resDf = pd.DataFrame()
        filePaths = self.get_all_file_paths(dataDir)
        for filename in filePaths:
            if not os.path.basename(filename).startswith(filename_prefix):
                continue
            df = pd.read_csv(filename, index_col=0)
            resDf = pd.concat([resDf, df], ignore_index = True)
        self.process_all_df(resDf)
        resDf.to_csv(dataDir + filename_res)
        print "Overall gap statistics: \n{}".format(resDf.describe())
        return
    def process_all_df(self, df):
        pass
    def add_timeid_col(self, df):
        df['time_id'] = df['time_slotid'].apply(singletonTimeslot.getTimeId)
        return
    def add_timedate_col(self, df):
        df['time_date'] = df['time_slotid'].apply(singletonTimeslot.getDate)
        return
    def sort_by_time(self, df):
        df['timeslotrank'] = df['time_slotid'].map(lambda x: "-".join(x.split('-')[:3] + [x.split('-')[-1].zfill(3)]))
        df.sort_values(by = ['timeslotrank'], inplace = True)
        df.drop('timeslotrank', axis = 1, inplace = True)
        df.reset_index(drop=True, inplace = True)
        return df
    def save_all_csv(self, dataDir):
        filePaths = self.get_all_file_paths(dataDir)
        for filename in filePaths:
            print "save csv for :{}".format(filename)
            self.save_one_csv(filename)
        return
    def save_one_csv(self, filename):
        print("loading file {}".format(filename))
        df = pd.read_csv(filename, delimiter='\t', header=None, names =['Time','weather','temparature', 'pm25'])
        df['time_slotid'] = df['Time'].map(singletonTimeslot.convertToSlot)
        
        self.process_one_df(df)
        
        df.reset_index(drop = True, inplace = True)
        df.to_csv(os.path.dirname(filename) + '/temp/'+ os.path.basename(filename) + '.csv')
        print df.describe()
        return df, filename
        return
    def process_one_df(self, df):
        pass