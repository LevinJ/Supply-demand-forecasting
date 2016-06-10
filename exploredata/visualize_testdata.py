from order import ExploreOrder
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(color_codes=True)
from utility.datafilepath import g_singletonDataFilePath
from visualization import visualizeData
import numpy as np
import math


class VisualizeTestData(visualizeData):
    def __init__(self):
        ExploreOrder.__init__(self)
        self.gapdf = self.load_gapdf(g_singletonDataFilePath.getTest1Dir())
        return
    def run(self):
        self.disp_gap_bytimeiid()
#         self.disp_gap_bydistrict()
#         self.disp_gap_bydate()
        return
    


if __name__ == "__main__":   
    obj= VisualizeTestData()
    obj.run()