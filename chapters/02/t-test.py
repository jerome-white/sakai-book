import sys
import csv
import math
import logging
import operator as op
from argparse import ArgumentParser

import numpy as np
import scipy.stats as st

import irstats as irs

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S')

def t_inv(phi, P):
    return st.t.ppf(1 - (1 - P) / 2, phi)

class T:
    def __init__(self, results, alpha):
        assert(results.ispaired())

        self.results = results
        self.alpha = alpha
        self.fieldnames = (
            'system_1',
            'system_2',
            'difference',
            't',
            'df',
            'p-value',
            'reject',
            'left_ci',
            'right_ci',
        )

    def __iter__(self):
        for i in self.results.systems():
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
            inverse = t_inv(df, self.alpha)
            reject = int(t0_ >= inverse)
            p = st.t.sf(t0_, df) * 2

            moe = inverse * variance
            ci = [ f(dbar, moe) for f in (op.sub, op.add) ]

            output = (*i.keys(), difference, t0, df, p, reject, *ci)
            yield dict(zip(self.fieldnames, output))

arguments = ArgumentParser()
arguments.add_argument('--alpha', type=float, default=0.95)
args = arguments.parse_args()

assert(0 <= args.alpha <= 1)

results = irs.Results.from_csv(sys.stdin)
t = T(results, args.alpha)

writer = csv.DictWriter(sys.stdout, fieldnames=t.fieldnames)
writer.writeheader()
writer.writerows(t)
