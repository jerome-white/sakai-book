import math
import operator as op
import itertools as it
import collections as cl
import multiprocessing as mp

import numpy as np
import scipy.stats as st

from irstats import inverse as irs
from .ci import ConfidenceInterval

Result = cl.namedtuple('Result', 'system1, system2, difference, p')
CompleteResult = cl.namedtuple('CompleteResult', Result._fields + ('reject', ))

def partitions(n, m):
    (x, y) = [ f(m, n) for f in (op.floordiv, op.mod) ]
    assert(n > y)
    pairs = it.zip_longest(it.repeat(x, n), it.repeat(1, y), fillvalue=0)

    yield from it.starmap(op.add, pairs)

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

class Tukey:
    def __init__(self, scores, alpha, Anova):
        self.scores = scores
        self.alpha = alpha
        self.anova = Anova(self.scores, alpha)

    def __iter__(self):
        V = float(self.anova.E)
        q = irs.q_inv(self.anova.m, self.anova.phiE, self.alpha) / math.sqrt(2)

        for i in self.scores.combinations():
            (x1, x2) = i.values()
            diff = np.subtract(*map(np.mean, (x1, x2)))

            normv = math.sqrt(V * sum([ 1 / len(x) for x in (x1, x2) ]))
            t = abs(diff / normv)
            reject = int(t >= q)
            p = st.t.sf(t, self.anova.m) * 2 # ???

            result = CompleteResult(*i.keys(), diff, p, reject)

            ci = ConfidenceInterval(diff, q * normv)

            yield { **result._asdict(), **ci.asdict() }

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
            iterable = partitions(self.workers, self.B)
            for i in pool.imap_unordered(self.do, iterable):
                self.c.update(i)

        return self

    def __next__(self):
        if not self.c:
            raise StopIteration()

        (k, v) = self.c.popitem()

        return Result(*k, self.d[k], v / self.B)

    def do(self, b):
        c = cl.Counter()

        for _ in range(b):
            x = next(self.shuffle)
            d = x.max() - x.min()

            for i in self.scores.combinations():
                systems = tuple(i.keys())
                if d >= abs(self.d[systems]):
                    c[systems] += 1

        return c
