from datetime import datetime
from datetime import timedelta

class PrepareHoldoutSet(object):
    def __init__(self):
        return
    def __get_date(self, start_date, days_num):
        startDate = datetime.strptime(start_date, '%Y-%m-%d')
        
        res = []
        for i in range(days_num):
            deltatime = timedelta(days = 2*i)
            item = (startDate + deltatime).date()
            res.append(str(item))
        return res
    def __get_slots(self):
        res = [46,58,70,82,94,106,118,130,142]
        return res
    def __unit_test(self):
        assert ['2016-01-13','2016-01-15','2016-01-17','2016-01-19','2016-01-21'] == self.__get_date('2016-01-13', 5)
        assert ['2016-01-12','2016-01-14','2016-01-16','2016-01-18','2016-01-20'] == self.__get_date('2016-01-12', 5)
        self.get_holdoutset(holdout_id = 1)
        
        assert ['2016-01-13-46','2016-01-13-58','2016-01-13-70','2016-01-13-82','2016-01-13-94','2016-01-13-106','2016-01-13-118','2016-01-13-130','2016-01-13-142'] == self.get_holdoutset(holdout_id = 101)
        print "unit test passed"
        return
    def get_holdoutset(self, holdout_id = 1):
        res = []
        start_date = '2016-01-13'
        days_num = 5
        if (holdout_id == 101):
            days_num = 1
        
        for d in self.__get_date(start_date, days_num):
            for s in self.__get_slots():
                res.append(d + '-' + str(s))
        return res
    def run(self):
        self.__unit_test()
        return



if __name__ == "__main__":   
    obj= PrepareHoldoutSet()
    obj.run()