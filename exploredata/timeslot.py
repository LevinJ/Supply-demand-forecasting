from datetime import datetime
from datetime import timedelta

class Timeslot:
    """Utility class for converting time slot ID
    one day is uniformly divided into 144 time slots t1,t2, t144, each 10 minutes long
    """
    def __init__(self):
        return
    def convertToSlot(self,strTime):
#         date_object = datetime.strptime(strTime, '%Y-%m-%d %H:%M:%S')
        hour = int(strTime[11:13])
        minute = int(strTime[14:16])
        timeslot = hour * 6 + (minute/10) + 1
        return strTime[:10] + "-"+ str(timeslot)
    def convertToStr(self, timeslotID):
        timeslotID = int(timeslotID[11:])
        initialtime = datetime.strptime('2016-01-03 00:00:00', '%Y-%m-%d %H:%M:%S')
        deltatime1 = timedelta(hours = timeslotID/6, minutes = (timeslotID%6 - 1)* 10)
        deltatime2 = timedelta(hours = timeslotID/6, minutes = (timeslotID%6 - 1)* 10 + 9, seconds = 59)
        starttime = initialtime + deltatime1
        endtime = initialtime + deltatime2
        return str(starttime.time()) +"--"+ str(endtime.time())
    def dispTimerange(self):
        self.convertToStr('2016-11-03-1')
        for i in range(145):
            if i == 0:
                continue
            timeslotID = '2016-11-03-'+ str(i)
            print i, self.convertToStr(timeslotID)
        return
    def convertToDateTime(self, timeslotID):
        tid = int(timeslotID[11:])
        initialtime = datetime.strptime(timeslotID[:10], '%Y-%m-%d')
        deltatime1 = timedelta(hours = tid/6, minutes = (tid%6 - 1)* 10 + 5)
        res = initialtime + deltatime1
        return res
    def getPrevSlots(self, timeslotID, preNum):
        timepoint = self.convertToDateTime(timeslotID)
        res = []
        for i in range(preNum):
            item = timepoint - timedelta(minutes = (i + 1) * 10)
            item = self.convertToSlot(str(item))
            res.append(item)
        return res
    def getTimeId(self, timeslotID):
        return int(timeslotID[11:])
    def getDate(self, timeslotID):
        return timeslotID[:10]
    def run(self):
        assert  "2016-01-03-125" == self.convertToSlot('2016-01-03 20:42:30'), "conversion error"
        assert  "2016-11-03-144" == self.convertToSlot('2016-11-03 23:59:30'), "conversion error"
        assert  "2009-01-03-1" == self.convertToSlot('2009-01-03 00:09:30'), "conversion error"
        assert  "00:00:00--00:09:59"==self.convertToStr('2016-11-03-1'), "Acutal " + self.convertToStr(1)
        assert  "23:50:00--23:59:59"==self.convertToStr('2016-11-03-144'), "Acutal " + self.convertToStr(144)
        assert  "07:30:00--07:39:59"==self.convertToStr('2016-11-03-46'), "Acutal " + self.convertToStr(46)
        
        assert  datetime.strptime('2016-11-03 00:05:00', '%Y-%m-%d %H:%M:%S')==self.convertToDateTime('2016-11-03-1')
        assert  datetime.strptime('2016-11-03 23:55:00', '%Y-%m-%d %H:%M:%S')==self.convertToDateTime('2016-11-03-144')
        assert  datetime.strptime('2016-11-03 07:35:00', '%Y-%m-%d %H:%M:%S')==self.convertToDateTime('2016-11-03-46')
        
        assert ['2016-01-10-1','2016-01-09-144' ,'2016-01-09-143' ] == self.getPrevSlots('2016-01-10-2', 3)
        assert ['2016-01-31-144','2016-01-31-143' ,'2016-01-31-142' ] == self.getPrevSlots('2016-02-01-1', 3)
        assert ['2015-12-31-144','2015-12-31-143' ,'2015-12-31-142', '2015-12-31-141' ] == self.getPrevSlots('2016-01-01-1', 4)
        
        assert 142 == self.getTimeId('2015-12-31-142')
        assert 1 == self.getTimeId('2016-11-03-1')
        assert 12 == self.getTimeId('2015-12-31-12')
        
        assert '2015-12-31' == self.getDate('2015-12-31-142')
        assert '2016-11-03' == self.getDate('2016-11-03-1')
        assert '2015-12-31' == self.getDate('2015-12-31-12')
        
        assert ['2016-01-10-1'] == self.getPrevSlots('2016-01-10-2', 1)
        assert ['2016-01-31-144'] == self.getPrevSlots('2016-02-01-1', 1)
        assert ['2015-12-31-144'] == self.getPrevSlots('2016-01-01-1', 1)
        
        print "passed the unit test"
        self.dispTimerange()
        return


singletonTimeslot= Timeslot()

if __name__ == "__main__":   
    obj= Timeslot()
    obj.run()