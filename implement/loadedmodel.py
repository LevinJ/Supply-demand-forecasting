import sys
import os
sys.path.insert(0, os.path.abspath('..'))


from utility.dumpload import DumpLoad
from decisiontreemodel import DecisionTreeModel
from knnmodel import KNNModel

class LoadedModel:
    def __init__(self):
        return
    def loadmodel (self):
        filename = r'2016_06_11_08_31_01_estimator.pickle'
        filepath = r'C:\Users\jianz\workspace\sdfp\implement\logs\\' + filename
        dumpload = DumpLoad(filepath)
        self.model = dumpload.load()
        return
    def run(self):
        self.loadmodel()
        self.model.getTrainTestSet()
        self.model.test()

        return




if __name__ == "__main__":   
    obj= LoadedModel()
    obj.run()