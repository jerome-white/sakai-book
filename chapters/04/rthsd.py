import sys
import csv
import itertools as it
import collections as cl
import multiprocessing as mp
from pathlib import Path
from argparse import ArgumentParser

import numpy as np

import irstats as irs

Result = cl.namedtuple('Result', 'system1, system2, difference, p')

class Shuffler:
    def __init__(self, scores, agg=True):
        self.scores = scores.astable().values
        self.agg = agg

    def __iter__(self):
        return self

    def __next__(self):
        scores = np.apply_along_axis(np.random.permutation, 1, self.scores)
        if self.agg:
            scores = scores.mean(axis=0)

        return scores

class RandomisedTukey:
    def __init__(self, scores, B, workers=None):
        self.scores = scores
        self.B = B
        self.workers = mp.cpu_count() if not workers else workers

        self.shuffle = Shuffler(self.scores)

        self.d = {}
        for i in self.scores.combinations():
            k = tuple(i.keys())
            v = np.subtract(*map(np.mean, i.values()))
            self.d[k] = v

        self.c = cl.Counter(dict(map(lambda x: (x, 0), self.d.keys())))

    def __iter__(self):
        with mp.Pool(self.workers) as pool:
            for i in pool.imap_unordered(self.do, range(self.B)):
                self.c.update(i)

        return self

    def __next__(self):
        if not self.c:
            raise StopIteration()

        (k, v) = self.c.popitem()

        return Result(*k, self.d[k], v / self.B)

    def do(self, b):
        c = cl.Counter()

        x = next(self.shuffle)
        d = x.max() - x.min()

        for i in self.scores.combinations():
            systems = tuple(i.keys())
            if d >= abs(self.d[systems]):
                c[systems] += 1

        return c

arguments = ArgumentParser()
arguments.add_argument('--B', type=int, default=0)
arguments.add_argument('--workers', type=int)
args = arguments.parse_args()

assert(args.B > 0)

scores = irs.Scores.from_csv(sys.stdin)
writer = None

for i in RandomisedTukey(scores, args.B, args.workers):
    if writer is None:
        writer = csv.DictWriter(sys.stdout, fieldnames=i._fields)
        writer.writeheader()
    writer.writerow(i._asdict())
