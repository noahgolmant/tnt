import math
from . import meter
import numpy as np


class AverageValueMeter(meter.Meter):
    def __init__(self, save_per_epoch=True):
        super(AverageValueMeter, self).__init__()
        self.reset()
        self.val = 0
        self.save_per_epoch = save_per_epoch

    def add(self, value, n=1):
        if not self.save_per_epoch:
            self.save(value)

        self.val = value
        self.sum += value
        self.var += value * value
        self.n += n

        if self.n == 0:
            self.mean, self.std = np.nan, np.nan
        elif self.n == 1:
            self.mean, self.std = self.sum, np.inf
            self.mean_old = self.mean
            self.m_s = 0.0
        else:
            self.mean = self.mean_old + (value - n * self.mean_old) / float(self.n)
            self.m_s += (value - self.mean_old) * (value - self.mean)
            self.mean_old = self.mean
            self.std = math.sqrt(self.m_s / (self.n - 1.0))

    def value(self):
        return self.mean, self.std

    def reset(self):
        if self.save_per_epoch:
            self.save((self.mean, self.std))
        else:
            self.reset_counter()

        self.n = 0
        self.sum = 0.0
        self.var = 0.0
        self.val = 0.0
        self.mean = np.nan
        self.mean_old = 0.0
        self.m_s = 0.0
        self.std = np.nan
