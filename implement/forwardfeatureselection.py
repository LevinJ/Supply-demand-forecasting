import numpy as np
import pickle
import pandas as pd
from implement.gradientboostingmodel import GrientBoostingModel
from implement.knnmodel import KNNModel
from preprocess.splittrainvalidation import HoldoutSplitMethod
from implement.xgboostmodel import DidiXGBoostModel

class ForwardFeatureSel:
    def __init__(self):
        clfDict = {1: GrientBoostingModel, 2:KNNModel, 3: DidiXGBoostModel}
        self.clf =  clfDict[3]()
        self.result = []
#         self.featureList = [101,102]
        self.featureList =  [101,102,103,104,105,106,107, 
                             201, 203,204,205,206,
                             301,
                             401,402,
                             501,502,503,504,505,506,507,
                             601,602,603,604,605,606,
                             8801,8802
                             ]

        return
    
    def selectBestFeaturList(self, featureLists):
        for fealist in featureLists:
                try:
                    self.clf.usedFeatures = fealist
                    print "Feature combination: {}".format(self.clf.get_used_features())

                    res =  self.clf.run_croos_validation() 
                    self.result.append([res, fealist, self.clf.get_used_features()])
                except Exception as inst:
                    print "XXXXXXX Ignore this combinationXXXXXX", inst, "featues used:  ", fealist
        self.disResult()
        return
    def run(self):
        base_feature_list = [101, 602, 402] 
        featureLists =  self.generateFeatureList(base_feature_list)
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
        df = pd.DataFrame(self.result, columns= ['scores', 'featlist_id','feature_list'])
        df2 = df.sort(columns = ['scores'], ascending=False).reset_index(drop = True)
        print df2
        print "@@@@@@@best combinations: ", df2[0:1].values
        return
    
#     print test

if __name__ == '__main__':
    test = ForwardFeatureSel() 
    test.run()