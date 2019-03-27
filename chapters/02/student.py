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

class TwoSample:
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
            logging.debug(st.ttest_ind(*i.values(), equal_var=True))

            x = i.values()
            (n1, n2) = map(len, x)
            xbar = [ np.mean(i) for i in x ]
            S = [ sum(np.square(op.sub(*i))) for i in zip(x, xbar) ]

            df = n1 + n2 - 2

            Vp = sum(S) / df
            t0 = op.sub(*xbar) / math.sqrt(Vp * (1 / n1 + 1 / n2))

            t0_ = abs(t0)
            reject = int(t0_ >= t_inv(df, self.alpha))
            p = st.t.sf(t0_, df) * 2

            output = (*i.keys(), *xbar, t0, df, p, reject)
            yield dict(zip(self.fieldnames, output))

arguments = ArgumentParser()
arguments.add_argument('--alpha', type=float, default=0.95)
args = arguments.parse_args()

assert(0 <= args.alpha <= 1)

results = irs.Results.from_csv(sys.stdin)
t = TwoSample(results, args.alpha)

writer = csv.DictWriter(sys.stdout, fieldnames=t.fieldnames)
writer.writeheader()
writer.writerows(t)
