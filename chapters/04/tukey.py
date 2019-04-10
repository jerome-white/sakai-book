import sys
import csv
import math
import collections as cl
from argparse import ArgumentParser

import numpy as np

# from irstats import Scores, OneWay, TwoWay
import irstats as irs

Result = cl.namedtuple('Result', 'system1, system2, difference, t, reject')

class Tukey:
    def __init__(self, scores, alpha):
        self.scores = scores
        self.alpha = alpha

        Anova = irs.TwoWay if self.scores.ispaired() else irs.OneWay
        self.anova = Anova(self.scores, alpha)

    def __iter__(self):
        q = irs.q_inv(self.anova.m, self.anova.phiE, self.alpha) / math.sqrt(2)

        for i in self.scores.combinations():
            diff = np.subtract(*map(np.mean, i.values()))
            V = self.anova.phiE * sum([ 1 / len(x) for x in i.values() ])
            t = diff / math.sqrt(V)
            reject = int(abs(t) >= q)

            yield Result(*i.keys(), diff, t, reject)

arguments = ArgumentParser()
arguments.add_argument('--alpha', type=float, default=0.95)
args = arguments.parse_args()

assert(0 <= args.alpha <= 1)

scores = irs.Scores.from_csv(sys.stdin)
tukey = Tukey(scores, args.alpha)

writer = None
for i in tukey:
    if writer is None:
        writer = csv.DictWriter(sys.stdout, fieldnames=i._fields)
        writer.writeheader()
    writer.writerow(i._asdict())
