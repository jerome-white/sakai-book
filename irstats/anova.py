import math
import operator as op
import itertools as it
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

    @property
    def phiE(self):
        m = self.scores.df[self.level].value_counts().size
        n = self.scores.df[self.level_].value_counts().sum()

        return n - m

    def S(self):
        f = lambda x: len(x) * (x['score'].mean() - self.grand_mean) ** 2
        s = self.scores.df.groupby('system').apply(f).sum()
        phi = len(self.scores.df[self.level].unique()) - 1

        yield Subject(s, phi, 'between({})'.format(self.level))
