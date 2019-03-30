import math
import logging
import operator as op
import itertools as it

import numpy as np
import scipy.stats as st

import irstats as irs

class T:
    def __init__(self, scores, alpha):
        self.scores = scores
        self.alpha = alpha
        self.fieldnames = [
            'system_1',
            'system_2',
            't',
            'df',
            'p-value',
            'reject',
            'left_ci',
            'right_ci',
        ]

    def __iter__(self):
        for i in self.scores.combinations():
            yield dict(zip(self.fieldnames, self.test(i)))

class Paired(T):
    def __init__(self, scores, alpha):
        assert(scores.ispaired())
        super().__init__(scores, alpha)

        self.fieldnames.extend([
            'difference',
        ])

    def test(self, i):
        logging.debug(st.ttest_rel(*i.values()))

        dj = np.subtract(*i.values())
        n = len(dj)
        df = n - 1

        # Equations 2.4, 2.5, and 2.6, respectively
        dbar = np.mean(dj)
        Vd = sum(np.square(dj - dbar)) / df
        variance = math.sqrt(Vd / n)
        t0 = dbar / variance

        difference = op.sub(*map(np.mean, i.values()))

        t0_ = abs(t0)
        inverse = irs.t_inv(df, self.alpha)
        reject = int(t0_ >= inverse)
        p = st.t.sf(t0_, df) * 2

        moe = inverse * variance
        ci = [ f(dbar, moe) for f in (op.sub, op.add) ]

        return (*i.keys(), t0, df, p, reject, *ci, difference)

class Unpaired(T):
    def __init__(self, scores, alpha):
        # assert(not scores.ispaired())
        super().__init__(scores, alpha)

        self.fieldnames.extend([
            'mean_1',
            'mean_2',
        ])

class Student(Unpaired):
    def test(self, i):
        logging.debug(st.ttest_ind(*i.values(), equal_var=True))

        x = i.values()
        (n1, n2) = map(len, x)
        xbar = [ np.mean(y) for y in x ]
        S = [ sum(np.square(np.subtract(*y))) for y in zip(x, xbar) ]

        df = n1 + n2 - 2

        Vp = sum(S) / df
        variance = math.sqrt(Vp * (1 / n1 + 1 / n2))
        t0 = op.sub(*xbar) / variance

        t0_ = abs(t0)
        inverse = irs.t_inv(df, self.alpha)
        reject = int(t0_ >= inverse)
        p = st.t.sf(t0_, df) * 2

        moe = inverse * variance
        difference = op.sub(*xbar)
        ci = [ f(difference, moe) for f in (op.sub, op.add) ]

        return (*i.keys(), t0, df, p, reject, *ci, *xbar)

class Welch(Unpaired):
    def test(self, i):
        logging.debug(st.ttest_ind(*i.values(), equal_var=False))

        x = i.values()
        n = list(map(len, x))
        xbar = [ np.mean(j) for j in x ]

        V = []
        for (j, k, l) in zip(x, xbar, n):
            value = sum(np.square(np.subtract(j, k))) / (l - 1)
            V.append(value)

        V_n = sum(it.starmap(op.truediv, zip(V, n)))
        variance = math.sqrt(V_n)

        df = V_n ** 2
        df /= sum([ (j / k) ** 2 / (k - 1) for (j, k) in zip(V, n) ])

        tw0 = op.sub(*xbar) / variance

        tw0_ = abs(tw0)
        inverse = irs.t_inv(df, self.alpha)
        reject = int(tw0_ >= inverse)
        p = st.t.sf(tw0_, df) * 2

        moe = inverse * variance
        difference = op.sub(*xbar)
        ci = [ f(difference, moe) for f in (op.sub, op.add) ]

        return (*i.keys(), tw0, df, p, reject, *ci, *xbar)
