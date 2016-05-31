
class DataFilePath:
    def __init__(self):
        self.dataDir = '../data/citydata/season_1/'
        return
    def getOrderDir_Train(self):  
        return self.dataDir + 'training_data/order_data/'
    def getOrderDir_Test1(self):  
        return self.dataDir + 'test_set_1/order_data/'
    def getGapCsv_Train(self):
        return self.getOrderDir_Train() + 'temp/gap.csv'
    def getGapCsv_Test1(self):
        return self.getOrderDir_Test1() + 'temp/gap.csv'

g_singletonDataFilePath = DataFilePath()