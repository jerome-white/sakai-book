import sys
import csv
import math
import operator as op
import itertools as it
import collections as cl
from argparse import ArgumentParser

import numpy as np

import irstats as irs

arguments = ArgumentParser()
arguments.add_argument('--alpha', type=float, default=0.95)
args = arguments.parse_args()

assert(0 <= args.alpha <= 1)

systems = irs.VarianceSystems(sys.stdin)
inv = irs.QInverse(args.alpha, systems.phi(), systems.systems)
MOE = float(inv) * math.sqrt(systems.V() / systems.topics)

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
