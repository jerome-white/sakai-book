import sys
import csv
import math
import logging
import operator as op
import itertools as it
from argparse import ArgumentParser

import numpy as np
import scipy.stats as st

import irstats as irs

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S')

def t_inv(phi, P):
    return st.t.ppf((1 - P) / 2, phi)

class Welch:
    def __init__(self, results, alpha):
        # assert(not results.ispaired())

        self.results = results
        self.alpha = alpha
        self.fieldnames = (
            'system_1',
            'system_2',
            'mean_1',
            'mean_2',
            't',
            'df',
            'p-value',
            'reject',
        )

    def __iter__(self):
        for i in self.results.systems():
            logging.debug(st.ttest_ind(*i.values(), equal_var=False))

            x = i.values()
            n = list(map(len, x))
            xbar = [ np.mean(i) for i in x ]

            V = []
            for (j, k, l) in zip(x, xbar, n):
                variance = sum(np.square(np.subtract(j, k))) / (l - 1)
                V.append(variance)

            V_n = sum(it.starmap(op.truediv, zip(V, n)))

            df = V_n ** 2 / sum([ (j / k) ** 2 / (k - 1) for (j, k) in zip(V, n) ])

            tw0 = op.sub(*xbar) / math.sqrt(V_n)

            tw0_ = abs(tw0)
            reject = int(tw0_ >= t_inv(df, self.alpha))
            p = st.t.sf(tw0_, df) * 2

            output = (*i.keys(), *xbar, tw0, df, p, reject)
            yield dict(zip(self.fieldnames, output))

arguments = ArgumentParser()
arguments.add_argument('--alpha', type=float, default=0.95)
args = arguments.parse_args()

assert(0 <= args.alpha <= 1)

results = irs.Results.from_csv(sys.stdin)
t = Welch(results, args.alpha)

writer = csv.DictWriter(sys.stdout, fieldnames=t.fieldnames)
writer.writeheader()
writer.writerows(t)
