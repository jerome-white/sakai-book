import csv
import itertools as it
import collections as cl

import pandas as pd

Score = cl.namedtuple('Score', 'system, topic, score')

class Scores:
    def __init__(self, scores):
        self.df = (pd
                   .DataFrame
                   .from_records(scores)
                   .pivot(index='topic', columns='system', values='score')
                   .astype(float))

    def __call__(self):
        return self.df

    def ispaired(self):
        return not self.df.isna().sum().any()

    def isgroupeq(self):
        return len(self.df.count().unique()) < 2

    def isreplicated(self):
        return self.df.index.duplicated.any()

    def combinations(self, n=2):
        for i in it.combinations(self.df.columns, r=n):
            j = [ self.df[x].tolist() for x in i ]
            yield dict(zip(i, j))

    def systems(self):
        return len(self.df.columns)

    def topics(self):
        if not self.ispaired():
            raise ValueError('Scores are not paired')

        return len(self.df.index.unique())

    def replicants(self):
        if not self.ispaired():
            raise ValueError('Scores are not paired')

        r = self.df.index.value_counts().unique().tolist()
        if len(r) > 1:
            raise ValueError('Replications not equal')

        return r.pop()

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
