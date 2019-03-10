import sys
import csv
import math
import operator as op
import itertools as it
import collections as cl
from argparse import ArgumentParser

import numpy as np
from scipy.stats import t

from irstats.systems import VarianceSystems

arguments = ArgumentParser()
arguments.add_argument('--confidence', type=float, default=0.95)
args = arguments.parse_args()

assert(0 <= args.confidence <= 1)

systems = VarianceSystems(sys.stdin)

#
# EXCEL::T.INV.2T(p, df) == t.ppf((1 + (1 - p)) / 2, df)
#
t_inv = t.ppf((1 + args.confidence) / 2, systems.topics)

MOE = t_inv * math.sqrt(systems.V() / systems.topics)

fieldnames = [
    'system_1',
    'system_2',
    'difference',
    'lower_ci',
    'upper_ci',
]
writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames)
writer.writeheader()
for (i, j) in systems.differences():
    #
    # Equation 4.19
    #
    interval = sorted([ f(j, MOE) for f in (op.sub, op.add) ])
    row = (*i, j, *interval)
    writer.writerow(dict(zip(fieldnames, row)))
