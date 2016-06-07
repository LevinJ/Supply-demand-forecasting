
import logging
import sys

class Logger:
    def __init__(self, **kwargs):
        logging.basicConfig(**kwargs)

        root = logging.getLogger()
        ch = logging.StreamHandler(sys.stdout)
        root.addHandler(ch)
        return
    



if __name__ == "__main__":   
    _=Logger(filename='logs/log.txt',filemode='w',level=logging.DEBUG)
    logging.debug('Here is a log sample {}'.format(1))