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

        axes = it.chain(range(2), (None, ))
        mean = lambda x: self.data.mean(axis=x)
        (self.s_mean, self.t_mean, self.m_mean) = map(mean, axes)

    #
    # Equation 5.16
    #
    def variance(self):
        #
        # S_E2
        #
        s = 0
        for (i, m) in enumerate(self.s_mean):
            x = self.m_mean - m
            for (j, n) in enumerate(self.t_mean):
                s += (self.data[(j, i)] - n + x) ** 2

        #
        # phi_E2
        #
        p = op.mul(*[ x - 1 for x in self.data.shape ])

        return s / p

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
    #
    # Equation 5.17
    #
    row = (*i, j / deviation)
    writer.writerow(dict(zip(fieldnames, row)))
