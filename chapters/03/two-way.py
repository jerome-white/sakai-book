import sys
import csv
from argparse import ArgumentParser

import irstats as irs
# from irstats import Scores

class TwoWay:
    def __init__(self, scores, alpha):
        # assert(not scores.ispaired())

        self.scores = scores
        self.alpha = alpha

        self.test = self.WR if self.scores.isreplicated() else self.WoR

    def WoR(self):
        s = self.scores()
        m = len(s.columns)

        xbar = s.sum().sum() / s.count().sum()
        ST = s.apply(lambda x: (x - xbar) ** 2, axis='columns').sum().sum()
        SA = (s
              .apply(lambda x: len(x) * (x.mean() - xbar) ** 2, axis='rows')
              .sum())
        SE1 = ST - SA

        SB = s.apply(lambda x: (x.mean() - xbar) ** 2, axis='columns').sum()
        SB *= m

        SE2 = SE1 - SB

        phiA = m - 1
        phiB = len(s) - 1
        phiE2 = (m - 1) * (n - 1)

        VA = SA / phiA
        VB = SB / phiB
        VE2 = SE2 / phiE2

        F0 = VA / VE2
        system_reject = int(F0 >= irs.F_inv(self.alpha, phiA, phiE2))

        F0_ = VB / VE2
        topic_reject = int(F0 >= irs.F_inv(self.alpha, phiB, phiE2))

arguments = ArgumentParser()
arguments.add_argument('--alpha', type=float, default=0.95)
args = arguments.parse_args()

assert(0 <= args.alpha <= 1)

results = Scores.from_csv(sys.stdin)
t = Paired(results, args.alpha)

writer = csv.DictWriter(sys.stdout, fieldnames=t.fieldnames)
writer.writeheader()
writer.writerows(t)
