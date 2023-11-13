from streamer.loggers.logger import Logger
import torch

class AudioLogger(Logger):


    def model(self, prediction, groundtruth):
        self.counter += 1

        # checks
        if (self.log_base_every<=0) or \
           (not len(self.layer_logs)) or \
           (self.counter%self.log_base_every != 0):
               return

        if isinstance(prediction, torch.Tensor):
            pass



