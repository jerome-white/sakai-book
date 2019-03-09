import sys
import csv
import math
import operator as op
import itertools as it
import collections as cl

import numpy as np

from irstats.systems import VarianceSystems

systems = VarianceSystems(sys.stdin)
MOE = math.sqrt(systems.V() / systems.topics)

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
