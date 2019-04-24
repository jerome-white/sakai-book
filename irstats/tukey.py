import math
import operator as op
import itertools as it
import collections as cl
import multiprocessing as mp

import numpy as np
import scipy.stats as st

from irstats import inverse as irs
from .ci import ConfidenceInterval
from .ttest import Effect

Result = cl.namedtuple('Result', 'system1, system2, difference, p, effect')
CompleteResult = cl.namedtuple('CompleteResult',
                               Result._fields + \
                               ('reject', ) + \
                               ConfidenceInterval.fields)

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

class TukeyEffect(Effect):
    def __init__(self, x1, x2, anova):
        super().__init__(x1, x2)

    def V(self):
        return self.anova.phiE

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

            effect = TukeyEffect(x1, x2, self.anova)
            ci = ConfidenceInterval(diff, q * normv)

            yield CompleteResult(*i.keys(),
                                 diff,
                                 p,
                                 effect,
                                 reject,
                                 **ci.asdict())

class RandomisedTukey:
    class Info:
        def __init__(self, difference, effect):
            self.difference = difference
            self.effect = effect
            self.count = 0

        def inc(self, counter):
            self.count += counter

    def __init__(self, scores, B, workers=None, baseline=None):
        self.scores = scores
        self.B = B
        self.workers = workers if workers else mp.cpu_count()

        self.shuffle = Shuffler(self.scores)
        self.info = {}

        if baseline is not None:
            baseline = self.scores[baseline].values

        for i in self.scores.combinations():
            key = tuple(i.keys())

            difference = np.subtract(*map(np.mean, i.values()))

            (x1, x2) = i.values()
            if baseline is None:
                baseline = x1
            effect = Glass(x1, x2, baseline)

            self.info[d] = Info(difference, effect)

    def __iter__(self):
        with mp.Pool(self.workers) as pool:
            iterable = partitions(self.workers, self.B)
            for i in pool.imap_unordered(self.do, iterable):
                for (k, v) in i.items():
                    self.info[k].inc(v)

        return self

    def __next__(self):
        if not self.c:
            raise StopIteration()

        (k, info) = self.info.popitem()
        p = info.count / self.B
        e = float(info.effect)

        return Result(*k, info.difference, p, e)

    def do(self, b):
        c = cl.Counter()

        for (_, x) in zip(range(b), self.shuffle):
            d = x.max() - x.min()
            for i in self.scores.combinations():
                systems = tuple(i.keys())
                if d >= abs(self.info[systems].difference):
                    c[systems] += 1

        return c
