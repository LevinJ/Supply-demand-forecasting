from sklearn.cross_validation import KFold


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
    def run(self, df):
        self.kfold_bydate(df)
        return
    

if __name__ == "__main__":   
    obj= SplitTrainValidation()
    from  preparedata import PrepareData
    from utility.datafilepath import g_singletonDataFilePath
    pre = PrepareData()
    pre.X_y_Df = pre.load_gapdf(g_singletonDataFilePath.getTrainDir())
    pre.transformXfDf(g_singletonDataFilePath.getTrainDir())
    obj.run(pre.X_y_Df)