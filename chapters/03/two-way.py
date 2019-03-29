import sys
import csv
from argparse import ArgumentParser

import irstats as irs
# from irstats import Scores

class Anova:
    def __init__(self, scores, alpha):
        # assert(not scores.ispaired())

        self.scores = scores
        self.alpha = alpha

        self.m = self.scores.systems()
        self.n = self.scores.topics()

class WithoutReplication(Anova):
    def test(self):
        s = self.scores()

        xbar = s.sum().sum() / s.count().sum()
        ST = s.apply(lambda x: (x - xbar) ** 2, axis='columns').sum().sum()
        SA = (s
              .apply(lambda x: len(x) * (x.mean() - xbar) ** 2, axis='rows')
              .sum())
        SB = (s
              .apply(lambda x: (x.mean() - xbar) ** 2, axis='columns')
              .sum()) * self.m

        SE1 = ST - SA
        SE2 = SE1 - SB

        phiA = self.m - 1
        phiB = self.n - 1
        phiE2 = phiA * phiB

        VA = SA / phiA
        VB = SB / phiB
        VE2 = SE2 / phiE2

        F0 = {}
        for (i, j, k) in zip(('system', 'topic'), (VA, VB), (phiA, phiB)):
            F = j / VE2
            F0[i] = (F, int(F >= irs.F_inv(k, phiE2, self.alpha)))

        VE2 = SE2 / phiE2
        MOE = irs.t_inv(phiE2, self.alpha) * math.sqrt(VE2 / self.n)
        ci = []
        for i in s:
            ci.append([ f(s[i].mean(), MOE) for f in (op.sub, op.add) ])

class WithReplication(Anova):
    def __init__(self, scores, alpha):
        assert(scores.isreplicated())
        super().__init__(scores, alpha)

        r = self.scores.replicants()

    def test(self):
        s = self.scores()

        xbar = s.mean().mean()
        ST = s.apply(lambda x: (x - xbar) ** 2, axis='columns').sum().sum()
        SA = (s
              .apply(lambda x: (x.mean() - xbar) ** 2, axis='rows')
              .sum()) * self.n * self.r
        SB = (s
              .apply(lambda x: (x.mean() - xbar) ** 2, axis='columns')
              .sum()) * self.m * self.r

        SE3 = (s
               .reset_index()
               .groupby('topic')
               .apply(lambda x: (x + 1) ** 2)
               .sum())

        SAxB = ST - SA - SB - S3

        phiA = self.m - 1
        phiB = self.n - 1
        phiAxB = phiA - phiB
        phiE3 = self.m * self.n * (self.r - 1)

        VA = SA / phiA
        VB = SB / phiB
        VAxB = SAxB / phiAxB
        V3 = SE3 / phiE3

        F0 = {}
        for (i, j, k) in zip(('system-topic', 'system', 'topic'),
                             (VAxB, VA, VB),
                             (phiAxB, phiA, phiB)):
            F = j / VE3
            F0[i] = (F, int(F >= irs.F_inv(k, phiE3, self.alpha)))

arguments = ArgumentParser()
arguments.add_argument('--alpha', type=float, default=0.95)
args = arguments.parse_args()

assert(0 <= args.alpha <= 1)

results = Scores.from_csv(sys.stdin)
t = Paired(results, args.alpha)

writer = csv.DictWriter(sys.stdout, fieldnames=t.fieldnames)
writer.writeheader()
writer.writerows(t)
