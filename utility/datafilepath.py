
class DataFilePath:
    def __init__(self):
        self.dataDir = '../data/citydata/season_1/'
        return
    def getOrderDir_Train(self):  
        return self.dataDir + 'training_data/order_data/'
    def getOrderDir_Test1(self):  
        return self.dataDir + 'test_set_1/order_data/'
    def getTest1Dir(self):
        return self.dataDir + 'test_set_1/'
    def getTest2Dir(self):
        return self.dataDir + 'test_set_2/'
    def getTrainDir(self):
        return self.dataDir + 'training_data/'
    def getGapCsv_Train(self):
        return self.getOrderDir_Train() + self.getGapFilename()
    def getGapCsv_Test1(self):
        return self.getOrderDir_Test1() + self.getGapFilename()
    def getTestset1Readme(self):
        return self.dataDir + 'test_set_1/read_me_1.txt'
    def getTestset2Readme(self):
        return self.dataDir + 'test_set_2/read_me_2.txt'
    def getGapFilename(self):
        return "temp/gap.csv"
    def getGapPredictionFileName(self):
        return 'gap_prediction.csv'
    def getPrevGapFileName(self):
        return "temp/prevgap.df.pickle"

g_singletonDataFilePath = DataFilePath()