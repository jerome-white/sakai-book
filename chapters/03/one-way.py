import sys
import csv
import math
import logging
import operator as op
from argparse import ArgumentParser

import scipy.stats as st

import irstats as irs
# from irstats import Scores

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S')

class OneWay_:
    def __init__(self, scores, alpha):
        # assert(not scores.ispaired())

        self.scores = scores
        self.alpha = alpha

        self.m = self.scores.systems()
        self.n = self.scores.topics()

    def test(self):
        s = self.scores()

        xbar = s.sum().sum() / s.count().sum()
        ST = s.apply(lambda x: (x - xbar) ** 2, axis='columns').sum().sum()
        SA = (s
              .apply(lambda x: len(x) * (x.mean() - xbar) ** 2, axis='rows')
              .sum())
        SE1 = ST - SA

        phiA = self.m - 1
        phiE1 = s.count().sum() - self.m

        F0 = (SA / phiA) / (SE1 / phiE1)
        reject = int(F0 >= irs.F_inv(phiA, phiE1, self.alpha))
        p = st.f.sf(F0, phiA, phiE1)

        VE1 = SE1 / phiE1
        MOE = irs.t_inv(phiE1, self.alpha) * math.sqrt(VE1 / self.n)
        ci = []
        for i in s:
            ci.append([ f(s[i].mean(), MOE) for f in (op.sub, op.add) ])

        logging.debug({
            'm': self.m,
            'n': self.n,
            'phiA': phiA,
            'phiE1': phiE1,
        })
        logging.info({
            'ST': ST,
            'SE1': SE1,
            'SA': SA,
            'VA': SA / phiA,
            'VE1': VE1,
            'F0': F0,
            'reject': reject,
            'p-value': p,
        })

arguments = ArgumentParser()
arguments.add_argument('--alpha', type=float, default=0.95)
args = arguments.parse_args()

assert(0 <= args.alpha <= 1)

results = irs.Scores.from_csv(sys.stdin)
# t = OneWay(results, args.alpha)
# t.test()

writer = None
for i in irs.OneWay(results, args.alpha):
    if writer is None:
        writer = csv.DictWriter(sys.stdout, fieldnames=i._fields)
        writer.writeheader()
    writer.writerow(i._asdict())
