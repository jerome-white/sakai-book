import operator as op
import collections as cl

class ConfidenceInterval:
    def __init__(self, mean, MOE):
        self.mean = mean
        self.MOE = MOE

        self.f = (op.sub, op.add)
        self.itr = None

    def __iter__(self):
        self.itr = iter(self.f)
        return self

    def __next__(self):
        return next(self.itr)(self.mean, self.MOE)
