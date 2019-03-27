import csv
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
    def __init__(self, scores):
        self.df = (pd
                   .DataFrame
                   .from_records(scores)
                   .pivot(index='topic', columns='system', values='score')
                   .astype(float))

    def ispaired(self):
        return not self.df.isna().sum().any()

    def isgroupeq(self):
        return len(self.df.count().unique()) < 2

    def systems(self, n=2):
        for i in it.combinations(self.df.columns, r=n):
            j = [ self.df[x].tolist() for x in i ]
            yield dict(zip(i, j))

    # def differences(self, stat):
    #     items = op.methodcaller(stat)(self.df).iteritems()
    #     for ((i, x), (j, y)) in it.combinations(items, r=2):
    #         yield PairVal(i, j, x - y)

    def sizes(self):
        yield from self.df.count().items()

    @classmethod
    def from_csv(cls, data):
        try:
            fp = open(data)
        except TypeError:
            fp = data

        scores = csv.DictReader(fp)
        assert(all([ x in scores.fieldnames for x in Score._fields ]))
        results = cls(scores)

        if fp != data:
            fp.close()

        return results
