import itertools as it

import numpy as np

class Systems:
    def __init__(self, fp):
        cols = fp.readline().rstrip().split(',')
        self.columns = { y: x for (x, y) in enumerate(cols) }
        self.data = np.loadtxt(fp, delimiter=',')
        # (self.runs, self.systems) = self.data.shape

    def __getitem__(self, key):
        index = self.columns[key]
        return self.data[:,index]

    def pairs(self):
        yield from it.combinations(self.columns, r=2)

    def differences(self):
        for i in self.pairs():
            yield (i, np.subtract(*[ self[x].mean() for x in i ]))
