import pandas as pd
from districtid import singletonDistricId

class ExplorePoi:
    """Utility class for converting time slot ID
    one day is uniformly divided into 144 time slots t1,t2, t144, each 10 minutes long
    """
    def __init__(self):
        
        return
    def __load_raw_poi(self):
        filename = '../data_raw/poi_data'
        raw_dict = {}
        with open(filename) as f:
            for line in f:
                self.__process_line(line, raw_dict)
        return raw_dict
    def __process_line(self, line, raw_dict):
        items = line.split('\t')
        line_dict_key = singletonDistricId.convertToId(items[0])
        raw_dict[line_dict_key] = {}
        line_dict = raw_dict[line_dict_key]
        items.pop(0)
        for item in items:
            try:
                item_key = int(item[:item.index('#')])
            except:
                item_key = int(item[:item.index(':')])
            item_value = int(item[item.index(':')+1:])
            if not item_key in line_dict:
                line_dict[item_key] = 0
            line_dict[item_key] = line_dict[item_key] + item_value
        
        return
    
    def __get_district_type_list(self):
        raw_dict = self.__load_raw_poi()
        res = set()
        for _,line_value in raw_dict.iteritems():
            for district_type_key, _ in line_value.iteritems():
                res.add(district_type_key)
        return res
    def get_district_type_list(self):
        type_list = self.__get_district_type_list()
        type_list_str = ['district_type_' + str(i) for i in type_list]
        return type_list_str
    def get_district_type_dict(self):
        raw_dict = self.__load_raw_poi()
        type_list = self.__get_district_type_list()
        type_list_str = ['district_type_' + str(i) for i in type_list]
        for district, district_dict in raw_dict.iteritems():
            temp = []
            for item in type_list:
                try:
                    item_value = district_dict[item]
                except:
                    item_value = 0
                temp.append(item_value)
            raw_dict[district] = pd.Series(temp, index=type_list_str)
        return raw_dict
    def find_poi(self, series, poi_dict=None):
        start_district_id = series.iloc[0]
        return poi_dict[start_district_id]
    def run(self):
        self.get_district_type_dict()
        return

#     obj.convertToId(districtHash)
if __name__ == "__main__":  
    obj =  ExplorePoi()
    obj.run()


