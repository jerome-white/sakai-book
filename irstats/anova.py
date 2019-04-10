import math
# import logging
import itertools as it
import functools as ft
import collections as cl

import numpy as np
import scipy.stats as st

import irstats as irs
from .ci import ConfidenceInterval

Test = cl.namedtuple('Test', 'F, p, reject')
Effect = cl.namedtuple('Effect',
                       ('factor', 'df', 'ssq', 'msq') + Test._fields,
                       defaults=(None, ) * len(Test._fields))

def powerset(collection, empty=True, full=True):
    c = list(collection)
    for i in range(int(not empty), len(c) + int(full)):
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

class Replications:
    def __init__(self, scores):
        self.shape = scores.shape()
        if not self.shape.replication:
            raise ValueError('Irregular replications')

    def __call__(self):
        return self.shape.replication

    def __bool__(self):
        return self() > 1

class Anova:
    levels = set([ 'system', 'topic' ])

    def __init__(self, scores, alpha, e1='system'):
        if e1 not in self.levels:
            raise ValueError('Unrecognized level {}'.format(e1))

        self.scores = scores
        self.alpha = alpha

        self.e1 = e1
        self.e2 = self.levels.difference(set([self.e1])).pop()

        self.subjects = []
        self.grand_mean = self.scores.df['score'].mean()

        shape = self.scores.shape()
        (self.m, self.n) = [ getattr(shape, x) for x in (self.e1, self.e2) ]

        # total
        scores = self.scores.df['score']
        ST = np.sum(np.square(np.subtract(scores, self.grand_mean)))
        phi = len(self.scores.df) - 1
        self.subjects.append(Subject(ST, phi, 'total'))

        # between
        self.subjects.extend(self.S())

        # within
        between = it.islice(self.subjects, 1, len(self.subjects))
        SE = self.subjects[0].S - sum([ x.S for x in between ])
        self.subjects.append(Subject(SE, self.phiE, 'within'))

    def __iter__(self):
        VE = float(self.E)
        last = len(self.subjects) - 1

        for (i, s) in enumerate(self.subjects):
            if 0 < i < last:
                F = float(s) / VE
                reject = int(F >= irs.F_inv(s.phi, self.E.phi, self.alpha))
                p = st.f.sf(F, s.phi, self.E.phi)
                t = Test(F, p, reject)
            else:
                t = None

            yield s.effect(t)

    def desq(self, x):
        return len(x) * (x['score'].mean() - self.grand_mean) ** 2

    def ci(self):
        VE = float(self.E)
        MOE = irs.t_inv(self.phiE, self.alpha) * math.sqrt(VE / self.n)

        for (i, g) in self.scores.df.groupby(self.e1):
            yield (i, ConfidenceInterval(g['score'].mean(), MOE))

    def S(self):
        raise NotImplementedError()

    @property
    def E(self):
        return self.subjects[-1]

class OneWay(Anova):
    def S(self):
        s = self.scores.df.groupby(self.e1).apply(self.desq).sum()
        phi = len(self.scores.df[self.e1].unique()) - 1
        name = 'between({name})'.format(name=self.e1)

        yield Subject(s, phi, name)

    @property
    def phiE(self):
        n = self.scores.df[self.e2].value_counts().sum()
        return n - self.m

class TwoWay(Anova):
    def __init__(self, scores, alpha):
        self.replication = Replications(scores)
        super().__init__(scores, alpha)

    @property
    def phiE(self):
        phi = (self.m - 1) * (self.n - 1)
        if self.replication:
            phi *= self.replication() - 1

        return phi

    @ft.lru_cache(maxsize=128)
    def inner(self, keys):
        xij = 0

        for (k, v) in zip(self.levels, keys):
            view = self.scores.df[self.scores.df[k] == v]
            xij += view['score'].mean()

        return xij

    def S(self):
        for i in powerset(self.levels, False, bool(self.replication)):
            if len(self.levels) == len(i):
                s = 0
                for (keys, g) in self.scores.df.groupby(list(self.levels)):
                    score = g['score'].mean()
                    s += (score - self.inner(keys) + self.grand_mean) ** 2
            else:
                s = self.scores.df.groupby(list(i)).apply(self.desq).sum()
            s *= self.replication()

            phi = 1
            for j in i:
                phi *= len(self.scores.df[j].unique()) - 1

            name = 'between({name})'.format(name='x'.join(i))

            yield Subject(s, phi, name)
