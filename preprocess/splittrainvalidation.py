from sklearn.cross_validation import KFold
import numpy as np
from datetime import datetime
from datetime import timedelta
from enum import Enum

class HoldoutSplitMethod(Enum):
#     NONE = 1
    BYDATESLOT_RANDOM = 2
    IMITATE_PUBLICSET = 3
    KFOLD_BYDATE      = 4
    kFOLD_FORWARD_CHAINING = 5
    IMITTATE_TEST2_MIN = 6
    IMITTATE_TEST2_FULL = 7
    IMITTATE_TEST2_PLUS1 = 8
    IMITTATE_TEST2_PLUS2 = 9
    IMITTATE_TEST2_PLUS4 = 10
    IMITTATE_TEST2_PLUS6 = 1
    
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
    def __get_slots(self, split_method):
        slot_split_dict = {}
        slot_split_dict[HoldoutSplitMethod.IMITTATE_TEST2_MIN] = self.__get_slots_min()
        slot_split_dict[HoldoutSplitMethod.IMITTATE_TEST2_FULL] = self.__get_slots_full()
        slot_split_dict[HoldoutSplitMethod.IMITTATE_TEST2_PLUS1] = self.__getplusslots(plus_num = 1)
        slot_split_dict[HoldoutSplitMethod.IMITTATE_TEST2_PLUS2] = self.__getplusslots(plus_num = 2)
        slot_split_dict[HoldoutSplitMethod.IMITTATE_TEST2_PLUS4] = self.__getplusslots(plus_num = 4)
        slot_split_dict[HoldoutSplitMethod.IMITTATE_TEST2_PLUS6] = self.__getplusslots(plus_num = 6)
        
        return slot_split_dict[split_method]
    def __get_slots_min(self):
        res = [46,58,70,82,94,106,118,130,142]
        return res
    def __get_slots_full(self):
        res = [i+1 for i in range(144)]
        return res
    def __get_date_slots(self, dates, slots):
        return [d + '-' + str(s) for d in dates for s in slots]
    def __getplusslots(self, plus_num = 2):
        res = []
        min_slots = self.__get_slots_min()
        for item in min_slots:
            for i in range(plus_num+1):
                x_below = item - i
                x_above = item + i 
                if x_below <= 144 and x_below >= 1:
                    res.append(x_below)
                if x_above <= 144 and x_above >= 1:
                    res.append(x_above)
        return np.sort(list(set(res)))
    def __unit_test(self):
        assert ['2016-01-13','2016-01-15','2016-01-17','2016-01-19','2016-01-21'] == self.__get_date('2016-01-13', 5)
        assert ['2016-01-12','2016-01-14','2016-01-16','2016-01-18','2016-01-20'] == self.__get_date('2016-01-12', 5)
        print self.__getplusslots(2)
        print self.__getplusslots(4)
        print self.__getplusslots(6)
#         self.get_holdoutset(holdout_id = 1)
#         
#         assert ['2016-01-13-46','2016-01-13-58','2016-01-13-70','2016-01-13-82','2016-01-13-94','2016-01-13-106','2016-01-13-118','2016-01-13-130','2016-01-13-142'] == self.get_holdoutset(holdout_id = 101)
        print "unit test passed"
        return
    def __get_df_indexes(self, df, dateslots):
        return df[df['time_slotid'].isin(dateslots)].index

    
    def get_imitate_testset2(self,df, split_method = HoldoutSplitMethod.IMITTATE_TEST2_MIN):
        df.sort_values(by = ['time_date','time_id','start_district_id'], axis = 0, inplace = True)
        df.reset_index(drop=True, inplace = True)
        res = []
        # training 1-15, validation 16-21
        item = self.__get_train_validation_indexes(df, '2016-01-01', 15, split_method), self.__get_train_validation_indexes(df, '2016-01-16', 6)
        res.append(item)
        
        # training 1-16, validation 17-21
        item = self.__get_train_validation_indexes(df, '2016-01-01', 16, split_method), self.__get_train_validation_indexes(df, '2016-01-17', 5)
        res.append(item)
        
        # training 1-17, validation 18-21
        item = self.__get_train_validation_indexes(df, '2016-01-01', 17, split_method), self.__get_train_validation_indexes(df, '2016-01-18', 4)
        res.append(item)
        
        # training 1-18, validation 19-21
        item = self.__get_train_validation_indexes(df, '2016-01-01', 18, split_method), self.__get_train_validation_indexes(df, '2016-01-19', 3)
        res.append(item)
        
        # training 1-19, validation 19-21
        item = self.__get_train_validation_indexes(df, '2016-01-01', 19, split_method), self.__get_train_validation_indexes(df, '2016-01-20', 2)
        res.append(item)
        
        # training 1-20, validation 21
        item = self.__get_train_validation_indexes(df, '2016-01-01', 20, split_method), self.__get_train_validation_indexes(df, '2016-01-21', 1)
        res.append(item)
        return res
    def __get_train_validation_indexes(self,df, start_date, days_num, split_method = HoldoutSplitMethod.IMITTATE_TEST2_MIN):
        dates = self.__get_date(start_date, days_num, days_step=1)
        slots = self.__get_slots(split_method) 
        dates_slots = self.__get_date_slots(dates, slots)
        indexes = self.__get_df_indexes(df, dates_slots)
        return indexes
    def run(self, df):
        self.__unit_test()
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