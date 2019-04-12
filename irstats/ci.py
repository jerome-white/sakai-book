import operator as op
import collections as cl

class ConfidenceInterval:
    fields = ('left', 'right')

    def __init__(self, mean, MOE):
        self.mean = mean
        self.MOE = MOE

        self.f = dict(zip(ConfidenceInterval.fields, (op.sub, op.add)))
        self.itr = None

    def __iter__(self):
        self.itr = iter(self.f.values())
        return self

    def __next__(self):
        return next(self.itr)(self.mean, self.MOE)

    def asdict(self):
        return dict(zip(self.f.keys(), self))
