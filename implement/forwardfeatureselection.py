import numpy as np
import pickle
import pandas as pd
from implement.gradientboostingmodel import GrientBoostingModel
from implement.knnmodel import KNNModel
from preprocess.splittrainvalidation import HoldoutSplitMethod

class ForwardFeatureSel:
    def __init__(self):
        clfDict = {'1': GrientBoostingModel, '2':KNNModel}
        self.clf =  clfDict['1']()
        self.result = []
        self.featureList =  [101,102,103,4,5,6, 701,702,703,801,802,901,902,903,904,10,11,1201,1202,1203,1204,1205,1206]
#         self.featureList =  [1,9]
        self.cv_method =[HoldoutSplitMethod.IMITTATE_TEST2_MIN]
#         self.cv_method =[HoldoutSplitMethod.IMITTATE_TEST2_MIN, HoldoutSplitMethod.IMITTATE_TEST2_PLUS1, HoldoutSplitMethod.IMITTATE_TEST2_PLUS2,
#                          HoldoutSplitMethod.IMITTATE_TEST2_PLUS3,HoldoutSplitMethod.IMITTATE_TEST2_PLUS4,HoldoutSplitMethod.IMITTATE_TEST2_PLUS6,
#                          HoldoutSplitMethod.IMITTATE_TEST2_FULL]
        return
    
    def selectBestFeaturList(self, featureLists):
        for fealist in featureLists:
            for split in self.cv_method:
                try:
                    self.clf.usedFeatures = fealist
                    self.clf.holdout_split = split
                    if hasattr(self.clf, 'busedFeaturesTranslated'):
                        del self.clf.busedFeaturesTranslated
                    res =  self.clf.run_croos_validation() 
                    self.result.append([res, fealist, self.clf.usedFeatures,split])
                except Exception as inst:
                    print "XXXXXXX Ignore this combinationXXXXXX", inst, "featues used:  ", fealist
        self.disResult()
        return
    def run(self):
    
        featureLists =  self.generateFeatureList([1201,101,5,1203,901,6,11,1206,1202,902,10,801])
        self.selectBestFeaturList(featureLists)
        
        return
    def generateFeatureList(self, basefeature):
        res = []
        for fea in self.featureList:
            if fea in basefeature:
                continue
            tempList = []
            tempList.append(fea)
            res.append(basefeature + tempList)
        if(len(res) == 0):
            res = [basefeature]
        return res
    def disResult(self):
        df = pd.DataFrame(self.result, columns= ['scores', 'featlist_id','feature_list', 'split'])
        df2 = df.sort(columns = ['scores'], ascending=True).reset_index(drop = True)
        print df2
        print "@@@@@@@best combinations: ", df2[0:1].values
        return
    
#     print test

if __name__ == '__main__':
    test = ForwardFeatureSel() 
    test.run()