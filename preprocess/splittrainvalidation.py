from sklearn.cross_validation import KFold
import numpy as np
from datetime import datetime
from datetime import timedelta


class SplitTrainValidation(object):
    def __init__(self):
        return
    def kfold_bydate(self, df, n_folds = 10):
        df.sort_values(by = ['time_date','time_id','start_district_id'], axis = 0, inplace = True)
        df.reset_index(drop=True, inplace = True)
        kf = KFold(df.shape[0], n_folds= n_folds, shuffle=False)
        for train_index, test_index in kf:
            print("TRAIN:", train_index, "TEST:", test_index)
        return kf
    def kfold_forward_chaining(self, df):
        res = []
        df.sort_values(by = ['time_date','time_id','start_district_id'], axis = 0, inplace = True)
        df.reset_index(drop=True, inplace = True)
        
        fold_len = df.shape[0]/10
        #fold 1-2, 3
        item = np.arange(0,2*fold_len), np.arange(2*fold_len, 3*fold_len)
        res.append(item)
        #fold 1-3, 4
        item = np.arange(0,3*fold_len), np.arange(3*fold_len, 4*fold_len)
        res.append(item)
        #fold 1-4, 5
        item = np.arange(0,4*fold_len), np.arange(4*fold_len, 5*fold_len)
        res.append(item)
        #fold 1-5, 6
        item = np.arange(0,5*fold_len), np.arange(5*fold_len, 6*fold_len)
        res.append(item)
        #fold 1-6, 7
        item = np.arange(0,6*fold_len), np.arange(6*fold_len, 7*fold_len)
        res.append(item)
        #fold 1-7, 8
        item = np.arange(0,7*fold_len), np.arange(7*fold_len, 8*fold_len)
        res.append(item)
        #fold 1-8, 9
        item = np.arange(0,8*fold_len), np.arange(8*fold_len, 9*fold_len)
        res.append(item)
        #fold 1-9, 10
        item = np.arange(0,9*fold_len), np.arange(9*fold_len, 10*fold_len)
        res.append(item)
        return res
    def __get_date(self, start_date, days_num, days_step=2):
        startDate = datetime.strptime(start_date, '%Y-%m-%d')
        
        res = []
        for i in range(days_num):
            deltatime = timedelta(days = days_step*i)
            item = (startDate + deltatime).date()
            res.append(str(item))
        return res
    def __get_slots_speicific(self):
        res = [46,58,70,82,94,106,118,130,142]
        return res
    def __get_slots_full(self):
        res = [i+1 for i in range(144)]
        return res
    def __get_date_slots(self, dates, slots):
        return [d + '-' + str(s) for d in dates for s in slots]
    def __unit_test(self):
        assert ['2016-01-13','2016-01-15','2016-01-17','2016-01-19','2016-01-21'] == self.__get_date('2016-01-13', 5)
        assert ['2016-01-12','2016-01-14','2016-01-16','2016-01-18','2016-01-20'] == self.__get_date('2016-01-12', 5)
        self.get_holdoutset(holdout_id = 1)
        
        assert ['2016-01-13-46','2016-01-13-58','2016-01-13-70','2016-01-13-82','2016-01-13-94','2016-01-13-106','2016-01-13-118','2016-01-13-130','2016-01-13-142'] == self.get_holdoutset(holdout_id = 101)
        print "unit test passed"
        return
    def __get_df_indexes(self, df, dateslots):
        return df[df['time_slotid'].isin(dateslots)].index
    def imitate_testset2(self, df):
        res = []
        df.sort_values(by = ['time_date','time_id','start_district_id'], axis = 0, inplace = True)
        df.reset_index(drop=True, inplace = True)
        
        # train 1-12, 1-144, validation 13-21, specific
        dates_train = self.__get_date('2016-01-01', 12, days_step=1)
        slots_train = self.__get_slots_full()
        dates_slots_train = self.__get_date_slots(dates_train, slots_train)
        
        dates_validation = self.__get_date('2016-01-13', 5, days_step=2)
        slots_validation = self.__get_slots_speicific()
        dates_slots_validation = self.__get_date_slots(dates_validation, slots_validation)
        
        item = self.__get_df_indexes(df, dates_slots_train), self.__get_df_indexes(df, dates_slots_validation)
        res.append(item)
        #train 1-12, specific, vaidation 13-21, specific
        dates_train = self.__get_date('2016-01-01', 12, days_step=1)
        slots_train = self.__get_slots_speicific()
        dates_slots_train = self.__get_date_slots(dates_train, slots_train)
        
        dates_validation = self.__get_date('2016-01-13', 5, days_step=2)
        slots_validation = self.__get_slots_speicific()
        dates_slots_validation = self.__get_date_slots(dates_validation, slots_validation)
        
        item = self.__get_df_indexes(df, dates_slots_train), self.__get_df_indexes(df, dates_slots_validation)
        res.append(item)
        res.extend(self.imitate_testset2_forwardchaning(df))
      
        return res
    def imitate_testset2_forwardchaning(self, df):
        res = []
        # training 1-15, validation 16-21
        item = self.__get_specific_indexes(df, '2016-01-01', 15), self.__get_specific_indexes(df, '2016-01-16', 6)
        res.append(item)
        
        # training 1-16, validation 17-21
        item = self.__get_specific_indexes(df, '2016-01-01', 16), self.__get_specific_indexes(df, '2016-01-17', 5)
        res.append(item)
        
        # training 1-17, validation 18-21
        item = self.__get_specific_indexes(df, '2016-01-01', 17), self.__get_specific_indexes(df, '2016-01-18', 4)
        res.append(item)
        
        # training 1-18, validation 19-21
        item = self.__get_specific_indexes(df, '2016-01-01', 18), self.__get_specific_indexes(df, '2016-01-19', 3)
        res.append(item)
        
        # training 1-19, validation 19-21
        item = self.__get_specific_indexes(df, '2016-01-01', 19), self.__get_specific_indexes(df, '2016-01-20', 2)
        res.append(item)
        
        # training 1-20, validation 21
        item = self.__get_specific_indexes(df, '2016-01-01', 20), self.__get_specific_indexes(df, '2016-01-21', 1)
        res.append(item)
        
        
        return res
    def __get_specific_indexes(self,df, start_date, days_num):
        dates = self.__get_date(start_date, days_num, days_step=1)
        slots = self.__get_slots_speicific()
        dates_slots = self.__get_date_slots(dates, slots)
        indexes = self.__get_df_indexes(df, dates_slots)
        return indexes
    def run(self, df):
        self.imitate_testset2(df)
#         self.kfold_bydate(df)
#         self.kfold_forward_chaining(df)
        return
    

if __name__ == "__main__":   
    obj= SplitTrainValidation()
    from  preparedata import PrepareData
    from utility.datafilepath import g_singletonDataFilePath
    pre = PrepareData()
    pre.X_y_Df = pre.load_gapdf(g_singletonDataFilePath.getTrainDir())
    pre.transformXfDf(g_singletonDataFilePath.getTrainDir())
    obj.run(pre.X_y_Df)