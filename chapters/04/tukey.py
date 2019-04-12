import sys
import csv
import math
import logging
import collections as cl
from argparse import ArgumentParser

import numpy as np
import scipy.stats as st

# from irstats import Scores, OneWay, TwoWay
import irstats as irs
from irstats.ci import ConfidenceInterval

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S')

Result = cl.namedtuple('Result', 'system1, system2, difference, p, reject')

class Tukey:
    def __init__(self, scores, alpha, Anova):
        self.scores = scores
        self.alpha = alpha
        self.anova = Anova(self.scores, alpha)

    def __iter__(self):
        V = float(self.anova.E)
        q = irs.q_inv(self.anova.m, self.anova.phiE, self.alpha) / math.sqrt(2)

        for i in self.scores.combinations():
            (x1, x2) = i.values()
            diff = np.subtract(*map(np.mean, (x1, x2)))

            normv = math.sqrt(V * sum([ 1 / len(x) for x in (x1, x2) ]))
            t = abs(diff / normv)
            reject = int(t >= q)
            p = st.t.sf(t, self.anova.m) * 2 # ???

            result = Result(*i.keys(), diff, p, reject)

            ci = ConfidenceInterval(diff, q * normv)

            yield { **result._asdict(), **ci.asdict() }

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
