import sys
import csv
import math
from argparse import ArgumentParser

from statsmodels.stats.power import FTestAnovaPower

arguments = ArgumentParser()
arguments.add_argument('--fstat', type=float,
                       help='F-statistic (F0) reported')
arguments.add_argument('--m', type=int,
                       help='Number of groups')
arguments.add_argument('--n', type=int,
                       help='Sample size')
arguments.add_argument('--alpha', type=float, default=0.05,
                       help='Type I Error probability')
arguments.add_argument('--power', type=float, default=0.80,
                       help='Desired statistical power')
args = arguments.parse_args()


fhat = math.sqrt(args.fstat * (args.m - 1) / (args.m * (args.n - 1)))

kwargs = {
    'effect_size': fhat,
    'k_groups': args.m,
    'alpha': args.alpha,
}

fpow = FTestAnovaPower()

power = fpow.solve_power(nobs=args.n, **kwargs)
sample = math.ceil(fpow.solve_power(power=args.power, **kwargs))

row = [
    fhat,
    power,
    sample,
]

fieldnames = ('fhat', 'power', 'size')
writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames)
writer.writeheader()
writer.writerow(dict(zip(fieldnames, row)))
