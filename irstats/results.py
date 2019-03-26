import itertools as it
import collections as cl
from pathlib import Path

import pandas as pd

__all__ = [
    'Score',
    'Results',
]

Score = cl.namedtuple('Score', 'system, topic, score')
PairVal = cl.namedtuple('PairVal', 'system_1, system_2, value')

class Results:
    def __init__(self, data, metric):
        self.e = Evaluation('system', 'topic', metric)

        try:
            fp = open(data)
        except TypeError:
            fp = data

        self.df = pd.read_csv(fp, usecols=self.e)

        if fp is not None:
            fp.close()

    def _is(self, grouper, mark, check):
        baseline = None

        for (_, g) in self.df.groupby(grouper):
            ptr = mark(g)
            if baseline is None:
                baseline = ptr
            elif check(baseline, ptr):
                return False

        return True

    def ispaired(self):
        return self._is(self.e.topic,
                        lambda x: set(x[self.e.system]),
                        lambda x, y: x.symmetric_difference(y))

    def isegs(self):
        return self._is(self.e.system,
                        lambda x: len(x),
                        lambda x, y: x != y)

    def pairs(self):
        systems = self.df[self.e.system].unique()
        yield from it.combinations(systems, r=2)

    def differences(self):
        items = (self
                 .astable()
                 .mean()
                 .iteritems())

        for ((i, x), (j, y)) in it.combinations(items, r=2):
            yield PairVal(i, j, x - y)

    def astable(self):
        return (self.df.pivot(index=self.e.topic,
                              columns=self.e.system,
                              values=self.e.metric))

    def sizes(self):
        for (i, g) in self.df.groupby(self.e.system):
            yield (i, len(g))
