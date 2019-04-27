import sys
import csv
import math
from argparse import ArgumentParser

from statsmodels.stats.power import TTestPower, TTestIndPower

arguments = ArgumentParser()
arguments.add_argument('--tstat', type=float,
                       help='t-statistic (t0) reported')
arguments.add_argument('--n1', type=int)
arguments.add_argument('--n2', type=int)
arguments.add_argument('--tail', type=int, default=2,
                       help='Sides of the test')
arguments.add_argument('--alpha', type=float, default=0.05,
                       help='Type I Error probability')
arguments.add_argument('--power', type=float, default=0.80,
                       help='Desired statistical power')
args = arguments.parse_args()

if args.tail == 1:
    alternative = 'one'
elif args.tail == 2:
    alternative = 'two'
else:
    raise ValueError('Invalid tail: {}. Should be either 1 or 2'
                     .format(args.tail))

es = abs(args.tstat) * math.sqrt((args.n1 + args.n2) / (args.n1 * args.n2))
kwargs = {
    'effect_size': es,
    'alpha': args.alpha,
    'alternative': '{}-sided'.format(alternative)
}

tpow = TTestPower()
sample = math.ceil(tpow.solve_power(power=args.power, **kwargs))

if args.n1 == args.n2:
    power = tpow.solve_power(nobs=args.n1, **kwargs)
else:
    ratio = args.n2 / args.n1
    if args.tail == 1:
        kwargs['alternative'] = 'larger' # ???
    tpow = TTestIndPower()
    power = tpow.solve_power(nobs1=args.n1, ratio=ratio, **kwargs)

row = [
    es,
    power,
    sample,
]

fieldnames = ('eshat', 'power', 'size')
writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames)
writer.writeheader()
writer.writerow(dict(zip(fieldnames, row)))
