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
    def run(self):
        assert  "2016-01-03-125" == self.convertToSlot('2016-01-03 20:42:30'), "conversion error"
        assert  "2016-11-03-144" == self.convertToSlot('2016-11-03 23:59:30'), "conversion error"
        assert  "2009-01-03-1" == self.convertToSlot('2009-01-03 00:09:30'), "conversion error"
        assert  "00:00:00--00:09:59"==self.convertToStr('2016-11-03-1'), "Acutal " + self.convertToStr(1)
        assert  "23:50:00--23:59:59"==self.convertToStr('2016-11-03-144'), "Acutal " + self.convertToStr(144)
        assert  "07:30:00--07:39:59"==self.convertToStr('2016-11-03-46'), "Acutal " + self.convertToStr(46)
        print "passed the unit test"
        self.dispTimerange()
        return


singletonTimeslot= Timeslot()

if __name__ == "__main__":   
    obj= Timeslot()
    obj.run()