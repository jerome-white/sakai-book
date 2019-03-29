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

    def test(self):
        s = self.scores()
        
        xbar = s.sum().sum() / s.count().sum()
        ST = s.apply(lambda x: (x - xbar) ** 2, axis='columns').sum().sum()
        SA = (s
              .apply(lambda x: len(x) * (x.mean() - xbar) ** 2, axis='rows')
              .sum())
        SE1 = ST - SA

        m = len(s.columns)
        phiA = m - 1
        phiE1 = s.apply(lambda x: len(x) - m, axis='rows').sum()

        F0 = (SA / phiA) / (SE1 / phiE1)

        reject = int(F0 >= irs.F_inv(self.alpha, phiA, phiE1))

arguments = ArgumentParser()
arguments.add_argument('--alpha', type=float, default=0.95)
args = arguments.parse_args()

assert(0 <= args.alpha <= 1)

results = Scores.from_csv(sys.stdin)
t = Paired(results, args.alpha)

writer = csv.DictWriter(sys.stdout, fieldnames=t.fieldnames)
writer.writeheader()
writer.writerows(t)
