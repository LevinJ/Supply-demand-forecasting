
from basemodel import BaseModel
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

class KNNModel(BaseModel):
    def __init__(self):
        BaseModel.__init__(self)
        return
    def getTunedParamterOptions(self):
        parameters = {'n_neighbors':np.arange(4,31,2), 'p':[1, 2]}
        return parameters
    def setClf(self):
        self.clf = KNeighborsClassifier()
        return




if __name__ == "__main__":   
    obj= KNNModel()
    obj.run()