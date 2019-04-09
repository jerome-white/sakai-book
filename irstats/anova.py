import math
import operator as op
import itertools as it
import functools as ft
import collections as cl

import numpy as np
import scipy.stats as st

import irstats as irs

Test = cl.namedtuple('Test', 'F, p, reject')
Effect = cl.namedtuple('Effect',
                       ('factor', 'df', 'ssq', 'msq') + Test._fields,
                       defaults=(None, ) * len(Test._fields))

def powerset(collection: list, empty=True):
    c = list(collection)
    for i in range(int(not empty), len(c) + 1):
        yield from it.combinations(c, r=i)

class Subject:
    def __init__(self, S, phi, factors):
        self.S = S
        self.phi = phi
        self.factors = factors

    def __str__(self):
        return str(self.factors)

    def __float__(self):
        return self.S / self.phi

    def effect(self, test=None):
        e = Effect(str(self), self.phi, self.S, float(self))
        if test is not None:
            dtc = e._asdict()
            dtc.update(test._asdict())
            e = Effect(**dtc)

        return e

class Anova:
    levels = set([ 'system', 'topic' ])

    def __init__(self, scores, alpha):
        self.scores = scores
        self.alpha = alpha

        self.grand_mean = self.scores.df['score'].mean()

        scores = self.scores.df['score']
        ST = np.sum(np.square(np.subtract(scores, self.grand_mean)))
        phi = len(self.scores.df) - 1
        self.total = Subject(ST, phi, 'total')

    def __iter__(self):
        between = list(self.S())
        SE = self.total.S - sum([ x.S for x in between ])
        E = Subject(SE, self.phiE, 'within')

        yield self.total.effect()

        for i in between:
            F = float(i) / float(E)
            reject = int(F >= irs.F_inv(i.phi, E.phi, self.alpha))
            p = st.f.sf(F, i.phi, E.phi)
            t = Test(F, p, reject)

            yield i.effect(t)

        yield E.effect()

class OneWay(Anova):
    def __init__(self, scores, alpha, level='system'):
        if level not in self.levels:
            raise ValueError('Unrecognized level {}'.format(level))

        super().__init__(scores, alpha)

        self.level = level
        self.level_ = self.levels.difference(set([self.level])).pop()

    def S(self):
        f = lambda x: len(x) * (x['score'].mean() - self.grand_mean) ** 2
        s = self.scores.df.groupby(self.level).apply(f).sum()
        phi = len(self.scores.df[self.level].unique()) - 1
        name = 'between({name})'.format(name=self.level)

        yield Subject(s, phi, name)

    @property
    def phiE(self):
        m = self.scores.df[self.level].value_counts().size
        n = self.scores.df[self.level_].value_counts().sum()

        return n - m

class TwoWay(Anova):
    def __init__(self, scores, alpha):
        super().__init__(scores, alpha)
        self.shape = self.scores.shape()
        if not self.shape.replication:
            raise ValueError('Irregular replications')

        self.replication = self.shape.replication > 1

    @property
    def phiE(self):
        phi = (self.shape.system - 1) * (self.shape.topic - 1)
        if self.replication:
            phi *= self.shape.replication - 1

        return phi

    @ft.lru_cache(maxsize=128)
    def inner(self, keys):
        xij = 0

        for (k, v) in zip(self.levels, keys):
            view = self.scores.df[self.scores.df[k] == v]
            xij += view['score'].mean()

        return xij

    def outer(self, x):
        return len(x) * (x['score'].mean() - self.grand_mean) ** 2

    def S(self):
        for i in powerset(self.levels, False):
            if len(self.levels) == len(i):
                if not self.replication:
                    continue
                s = 0
                for (keys, g) in self.scores.df.groupby(list(self.levels)):
                    score = g['score'].mean()
                    s += (score - self.inner(keys) + self.grand_mean) ** 2
            else:
                s = self.scores.df.groupby(list(i)).apply(self.outer).sum()
            s *= self.shape.replication

            phi = 1
            for j in i:
                phi *= len(self.scores.df[j].unique()) - 1

            name = 'between({name})'.format(name='x'.join(i))

            yield Subject(s, phi, name)
