
from utility.sklearnbasemodel import BaseModel
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
from sklearn.pipeline import Pipeline
from preprocess.preparedata import HoldoutSplitMethod
import numpy as np

class KNNModel(BaseModel):
    def __init__(self):
        BaseModel.__init__(self)
        self.usedFeatures = [101,102,103,104,105,106,107, 
                             201, 203,204,205,206,
                             301,
                             401,402,
                             501,502,503,504,505,506,507,
                             601,602,603,604,605,606,
                             8801,8802
                             ]
        self.usedFeatures = [603, 101, 602, 103, 203, 606]
        self.train_validation_foldid = -2
#         self.save_final_model = True
        self.do_cross_val = False
        return
    def setClf(self):
        clf = KNeighborsClassifier(n_neighbors = 25)
        min_max_scaler = preprocessing.MinMaxScaler()
        self.clf = Pipeline([('scaler', min_max_scaler), ('estimator', clf)])
        return
    def getTunedParamterOptions(self):
#         tuned_parameters = [{'n_neighbors': np.arange(2, 150, 1)}]
        tuned_parameters = [{'estimator__n_neighbors': np.arange(30, 36, 1)}]
        return tuned_parameters




if __name__ == "__main__":   
    obj= KNNModel()
    obj.run()