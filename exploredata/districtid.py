import pandas as pd


class DistricId:
    """Utility class for converting time slot ID
    one day is uniformly divided into 144 time slots t1,t2, t144, each 10 minutes long
    """
    def __init__(self):
        filename = '../data_raw/cluster_map'
        df = pd.read_csv(filename, delimiter='\t', header=None, names =['distric_hash','district_id'])
        self.dictDistrict = {}
        for _, row in df.iterrows():
            self.dictDistrict[row[0]] = row[1]
        return

    def convertToId(self, districtHash):
        return self.dictDistrict[districtHash]
    def run(self):
        assert 19 == self.convertToId('a5609739c6b5c2719a3752327c5e33a7'), self.convertToId('a5609739c6b5c2719a3752327c5e33a7')
        assert 66 == self.convertToId('1ecbb52d73c522f184a6fc53128b1ea1'), self.convertToId('1ecbb52d73c522f184a6fc53128b1ea1')
        print "passed unit test"
        return


singletonDistricId= DistricId()
# def convertToId(districtHash):
#     obj.convertToId(districtHash)
if __name__ == "__main__":  
    obj =  DistricId()
    obj.run()
