import sys
import csv
import math
import operator as op
import itertools as it
import collections as cl

import numpy as np

from irstats.systems import Systems

class ESSystems(Systems):
    def __init__(self, fp):
        super().__init__(fp)

        axes = it.chain((None, ), range(2))
        mean = lambda x: self.data.mean(axis=x)
        (self.m_mean, self.s_mean, self.t_mean) = map(mean, axes)

    def S(self):
        for (i, m) in enumerate(self.s_mean):
            x = self.m_mean - m
            for (j, n) in enumerate(self.t_mean):
                yield (self.data[(j, i)] - n + x) ** 2

    def phi(self):
        return op.mul(*[ x - 1 for x in self.data.shape ])

    def variance(self):
        return sum(self.S()) / self.phi()

systems = ESSystems(sys.stdin)
deviation = math.sqrt(systems.variance())

fieldnames = [
    'system_1',
    'system_2',
    'effect',
]
writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames)
writer.writeheader()
for (i, j) in systems.differences():
    row = (*i, j / deviation)
    writer.writerow(dict(zip(fieldnames, row)))
