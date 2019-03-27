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
    return st.t.ppf((1 - P) / 2, phi)

# def q_inv(P):
#     return st.norm.ppf(1 - P)

class TTest:
    def __init__(self, results, alpha):
        assert(results.isgroupeq())

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
            t0 = dbar / math.sqrt(Vd / n)

            difference = op.sub(*map(np.mean, i.values()))

            t0_ = abs(t0)
            reject = int(t0_ >= t_inv(df, self.alpha))
            p = st.t.sf(t0_, df) * 2

            output = (*i.keys(), difference, t0, df, p, reject)
            yield dict(zip(self.fieldnames, output))

arguments = ArgumentParser()
arguments.add_argument('--alpha', type=float, default=0.95)
args = arguments.parse_args()

assert(0 <= args.alpha <= 1)

results = irs.Results.from_csv(sys.stdin)
t = TTest(results, args.alpha)

writer = csv.DictWriter(sys.stdout, fieldnames=t.fieldnames)
writer.writeheader()
writer.writerows(t)
