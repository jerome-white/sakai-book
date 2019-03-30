import sys
import csv
from argparse import ArgumentParser

import irstats as irs
# from irstats import Scores

class OneWay:
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
        phiE1 = s.apply(lambda x: len(x) - self.m, axis='rows').sum()

        F0 = (SA / phiA) / (SE1 / phiE1)
        reject = int(F0 >= irs.F_inv(phiA, phiE1, self.alpha))

        VE1 = SE1 / phiE1
        MOE = irs.t_inv(phiE1, self.alpha) * math.sqrt(VE1 / self.n)
        ci = []
        for i in s:
            ci.append([ f(s[i].mean(), MOE) for f in (op.sub, op.add) ])

arguments = ArgumentParser()
arguments.add_argument('--alpha', type=float, default=0.95)
args = arguments.parse_args()

assert(0 <= args.alpha <= 1)

results = Scores.from_csv(sys.stdin)
t = Paired(results, args.alpha)

writer = csv.DictWriter(sys.stdout, fieldnames=t.fieldnames)
writer.writeheader()
writer.writerows(t)
