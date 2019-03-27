import sys
import csv
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

def q_inv(P):
    return st.norm.ppf(1 - P)

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

            d_j = np.subtract(*i.values())
            n = len(d_j)
            df = n - 1

            # Equations 2.4, 2.5, and 2.6, respectively
            d_bar = np.mean(d_j)
            V_d = sum(np.square(d_j - d_bar)) / df
            t_0 = d_bar / np.sqrt(V_d / n)

            difference = op.sub(*map(np.mean, i.values()))

            t_0_ = abs(t_0)
            reject = int(t_0_ >= t_inv(df, self.alpha))
            p = st.t.sf(t_0_, df) * 2

            output = (*i.keys(), difference, t_0, df, p, reject)
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
