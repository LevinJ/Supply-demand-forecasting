from sklearn.cross_validation import KFold
import numpy as np


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
    def run(self, df):
#         self.kfold_bydate(df)
        self.kfold_forward_chaining(df)
        return
    

if __name__ == "__main__":   
    obj= SplitTrainValidation()
    from  preparedata import PrepareData
    from utility.datafilepath import g_singletonDataFilePath
    pre = PrepareData()
    pre.X_y_Df = pre.load_gapdf(g_singletonDataFilePath.getTrainDir())
    pre.transformXfDf(g_singletonDataFilePath.getTrainDir())
    obj.run(pre.X_y_Df)