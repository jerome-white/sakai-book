import csv
import itertools as it
import collections as cl

import pandas as pd

Score = cl.namedtuple('Score', 'system, topic, score')
Shape = cl.namedtuple('Shape', Score._fields + ('replication', ))

class Scores:
    def __init__(self, scores):
        self.df = (pd
                   .DataFrame
                   .from_records(scores)
                   .astype({ 'score': float }))

    def __call__(self):
        return self.df

    def __getitem__(self, key):
        index = 'topic'
        return (self
                .df
                .query('system == @key')
                .filter(items=[index, 'scores'])
                .set_index(index)
                .squeeze())

    #
    #
    #

    def _size(self, factor):
        return self.df.groupby(factor, sort=False).apply(len).unique()

    def astable(self):
        return self.df.pivot(index='topic', columns='system', values='score')

    def combinations(self, n=2, aspect='system'):
        view = self.df[aspect]
        for i in it.combinations(view.unique(), r=n):
            values = [ self.df[view == x]['score'].tolist() for x in i ]
            yield dict(zip(i, values))

    def shape(self):
        keys = [ 'system', 'topic' ]
        systop = [ self.df[x].unique().size for x in keys ]

        sizes = self._size(keys)
        r = sizes.item() if len(sizes) == 1 else float('nan')

        return Shape(*systop, len(self.df), r)

    #
    # Data is "paired" if every system contains a score for every
    # topic.
    #
    def ispaired(self):
        topics = None

        for (_, g) in self.df.groupby('system', sort=False):
            t = sorted(g['topic'].tolist())
            if topics is None:
                topics = t
            elif topics != t:
                return False

        return True

    def isgroupeq(self):
        return self._size('system').size == 1

    # def isreplicated(self):
    #     return self.df.index.duplicated().any()

    #
    # Read data from a CSV stream
    #
    @classmethod
    def from_csv(cls, data):
        try:
            fp = open(data)
        except TypeError:
            fp = data

        scores = csv.DictReader(fp)
        if not all([ x in scores.fieldnames for x in Score._fields ]):
            raise ValueError('Invalid header: received: {}, expecting: {}'
                             .format(scores.fieldnames, Score._fields))
        results = cls(scores)

        if fp != data:
            fp.close()

        return results
