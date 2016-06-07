import datetime
import time
import dateutil.relativedelta
import logging

class Duration:
    def start(self):
        logging.debug('start timer')
        self.startTime = datetime.datetime.now()
        return
    def end(self):
        self.endTime = datetime.datetime.now()
        logging.debug('end timer')
        self.dispDuration()
        return
    def dispDuration(self):
        rd = dateutil.relativedelta.relativedelta (self.endTime , self.startTime)
        logging.debug( "Duration: %d hours, %d minutes and %d seconds" % (rd.hours, rd.minutes, rd.seconds))
#         logging.debug "Duration: %d years, %d months, %d days, %d hours, %d minutes and %d seconds" \
#         % (rd.years, rd.months, rd.days, rd.hours, rd.minutes, rd.seconds)
            
        
        return
    
if __name__ == "__main__":   
    obj= Duration()
    obj.start()
    time.sleep(3)
    obj.end()