import operator as op
import itertools as it

import numpy as np

class Systems:
    def __init__(self, fp):
        cols = fp.readline().rstrip().split(',')
        self.columns = { y: x for (x, y) in enumerate(cols) }
        self.data = np.loadtxt(fp, delimiter=',')
        (self.topics, self.systems) = self.data.shape

    def __getitem__(self, key):
        index = self.columns[key]
        return self.data[:,index]

    def pairs(self):
        yield from it.combinations(self.columns, r=2)

    def differences(self):
        for i in self.pairs():
            yield (i, np.subtract(*[ self[x].mean() for x in i ]))

class VarianceSystems(Systems):
    def __init__(self, fp):
        super().__init__(fp)

        axes = it.chain(range(2), (None, ))
        mean = lambda x: self.data.mean(axis=x)
        (self.s_mean, self.t_mean, self.m_mean) = map(mean, axes)

    def S(self):
        s = 0

        for (i, m) in enumerate(self.s_mean):
            x = self.m_mean - m
            for (j, n) in enumerate(self.t_mean):
                s += (self.data[(j, i)] - n + x) ** 2

        return s

    def phi(self):
        return op.mul(*[ x - 1 for x in self.data.shape ])

    def V(self):
        return self.S() / self.phi()
