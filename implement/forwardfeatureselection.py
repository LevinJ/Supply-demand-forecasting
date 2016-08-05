import numpy as np
import pickle
import pandas as pd
from implement.gradientboostingmodel import GrientBoostingModel
from implement.knnmodel import KNNModel
from preprocess.splittrainvalidation import HoldoutSplitMethod
from implement.xgboostmodel import DidiXGBoostModel
import logging
import sys

class ForwardFeatureSel:
    def __init__(self):
        root = logging.getLogger()
        root.setLevel(logging.DEBUG)
        root.addHandler(logging.StreamHandler(sys.stdout))
        root.addHandler(logging.FileHandler('logs/forwardfeatureselection.log', mode='w'))
        clfDict = {1: GrientBoostingModel, 2:KNNModel, 3: DidiXGBoostModel}
        self.clf =  clfDict[3]()
#         self.result = []
#         self.featureList = [101,102, 201,502]
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
        result_combinations = []
        for fealist in featureLists:
                try:
                    self.clf.usedFeatures = fealist
                    logging.info( "Feature combination: {}, {}".format(self.clf.usedFeatures, self.clf.get_used_features()))

                    res =  self.clf.run_croos_validation() 
                    result_combinations.append([res, fealist, self.clf.get_used_features()])
                except Exception as inst:
                    logging.info("XXXXXXX Ignore this combinationXXXXXX {}, features used: {}".format(inst,  fealist))
        return self.disResult(result_combinations)
        
    def run(self):
        base_feature_list = [] 
        featureLists =  self.generateFeatureList(base_feature_list)
        self.selectBestFeaturList(featureLists)
        
        return
    def auto_run(self):
        base_feature_list = []
        res = []
        while len(base_feature_list) < len(self.featureList):
            featureLists =  self.generateFeatureList(base_feature_list)
            base_feature_list, best_score = self.selectBestFeaturList(featureLists)
            res.append((best_score, base_feature_list, self.clf.get_used_features()))
        df = pd.DataFrame(res, columns=['scores', 'featlist_id', 'feature_list'])
        logging.info("Final Result: {}".format(df))
        
        df2 = df.sort(columns = ['scores'], ascending=False).reset_index(drop = True)
        logging.info("################Final Best combinations: {}".format(df2[0:1].values))
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
    def disResult(self, result_combinations):
        df = pd.DataFrame(result_combinations, columns= ['scores', 'featlist_id','feature_list'])
        df2 = df.sort(columns = ['scores'], ascending=False).reset_index(drop = True)
        logging.info("result for above feature combination: {}".format(df2))
        logging.info("@@@@@@@best combinations: {}".format(df2[0:1].values))
        return df2[0:1].featlist_id[0], df2[0:1].scores[0]
    

if __name__ == '__main__':
    test = ForwardFeatureSel() 
#     test.run()
    test.auto_run()