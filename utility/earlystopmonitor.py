
import logging

class EarlyStopMonitor(object):
    def __init__(self):
        self.early_stopping_rounds = None
        self.max_eval = None
        self.max_eval_step = None
        return
    def get_stop_decisision(self, current_step, current_eval):
#         logging.info("step {}, value {}".format(current_step, current_eval))
        if self.early_stopping_rounds is None:
            return False
        if self.max_eval is None:
            # This is the first current_step
            self.max_eval = current_eval
            self.max_eval_step = current_step
            return False
        if self.max_eval < current_eval:
            self.max_eval = current_eval
            self.max_eval_step = current_step
        step_gap = current_step - self.max_eval_step 
        if step_gap >= self.early_stopping_rounds:
            return True
        return False