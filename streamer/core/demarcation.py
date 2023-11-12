from collections import deque
import numpy as np


class EventDemarcation():
    r"""
    Define different Demarcation strategies

    :param str dem_mode: the type of demarcation loss: ['fixed', 'accum', 'average']
    :param str dist_mode: the type of demarcation loss: ['similarity', 'distance']
    :param float threshold: error or similarity threshold value. Only used with 'accum' and 'fixed' demarcation modes
    :param int window_size: Window size for moving average. Only used for 'average' demarcation mode
    :param str modifier_type: type of change to average: ['add', 'multiply']. Only used for 'average' demarcation mode
    :param float modifier: multiplier to change the average value as threshold. Only used for 'average' demarcation mode
    """

    def __init__(self, dem_mode, dist_mode, **kwargs):
        self.dem_mode = dem_mode
        self.dist_mode = dist_mode


        if dem_mode == 'fixed':
            self.threshold = kwargs['threshold']
        elif dem_mode == 'accum':
            self.buffer = 0.0
            self.threshold = kwargs['threshold']
            assert self.dist_mode == 'distance', "Use distance measure with accum demarcation type"
        elif dem_mode == 'average':
            self.window = deque(maxlen=kwargs['window_size'])
            self.modifier = kwargs['modifier']
            self.modifier_type =kwargs['modifier_type']
        else:
            raise NotImplementedError('dem mode %s not implemented' % dem_mode)

    def __call__(self, value):
        r"""
        Detect boundary according to demarcation mode and distance mode

        :param float value: The new error value
        :returns:
            (*bool*): True if the new value in a boundary

        """

        if self.dem_mode == 'fixed':
            if self.dist_mode == 'similarity': return (value < self.threshold)
            elif self.dist_mode == 'distance': return (value > self.threshold)

        elif self.dem_mode == 'accum':
            self.buffer += value

            if self.buffer > self.threshold:
                self.buffer = 0.0
                return True
            else: return False

        elif self.dem_mode == 'average':
            if len(self.window) == 0:
                self.window.append(value)
                return False

            avg = np.average(list(self.window))

            if self.modifier_type == 'add': avg += self.modifier
            elif self.modifier_type == 'multiply': avg *= self.modifier

            boundary = value<avg if self.dist_mode == 'similarity' else value > avg
            self.window.append(value)
            return boundary

    def reset(self):
        r"""
        Reset the window for demarcation. Only affects 'accum' and 'average' demarcation modes
        """

        if hasattr(self, 'window'):
            self.window.clear()

        if hasattr(self, 'buffer'):
            self.buffer = 0.0





