import sys
import csv
import math
import logging
import operator as op
import multiprocessing as mp
from argparse import ArgumentParser

import scipy.stats as st

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S')

def valid(args):
    focus = ('effect_size', 'difference', 'sigma')
    (a, *b) = [ bool(getattr(args, x)) for x in focus ]

    return a ^ op.or_(*b)

class Sample:
    def __init__(self, n, alpha, beta, delta):
        self.n = n
        self.beta = beta

        lamb = math.sqrt(self.n) * delta
        phi = self.n - 1
        w = st.t.ppf(1 - alpha / 2, phi)

        normalization = math.sqrt(1 + w ** 2 / (2 * phi))
        f = lambda x: (x * (1 - 1 / (4 * phi)) - lamb) / normalization
        self.pr = [ st.norm.cdf(f(g(w))) for g in (op.neg, op.pos) ]

    def __float__(self):
        (left, right) = self.pr

        return float(left + 1 - right)

    def __int__(self):
        return self.n

    def __bool__(self):
        return 1 - self.beta < float(self)

arguments = ArgumentParser()
arguments.add_argument('--alpha', type=float,
                       help='Type I Error probability')
arguments.add_argument('--beta', type=float,
                       help='Type II Error probability')
arguments.add_argument('--effect-size', type=float,
                       help='Minimum effect size')
arguments.add_argument('--difference', type=float,
                       help='Minimum detectable difference')
arguments.add_argument('--sigma', type=float,
                       help='Variance estimate for score differences')
arguments.add_argument('--radius', type=int, default=0)
arguments.add_argument('--workers', type=int, default=mp.cpu_count())
args = arguments.parse_args()

if not valid(args):
    raise ValueError('Must specify either an effect '
                     'size, or a difference and sigma, '
                     'but not both.')

with mp.Pool(args.workers) as pool:
    if args.effect_size is None:
        min_delta = args.difference / math.sqrt(args.sigma)
    else:
        min_delta = args.effect_size

    (zalpha, zbeta) = map(st.norm.ppf, (1 - args.alpha / 2, args.beta))
    n = ((zalpha - zbeta) / min_delta) ** 2 + zalpha ** 2 / 2

    logging.debug(n)

    recommended = math.ceil(n)
    start = max(2, recommended - args.radius)
    stop = recommended + args.radius + 1

    fieldnames = ('n', 'adequate', 'achieved')
    writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames)
    writer.writeheader()
    
    f = lambda x: (x, args.alpha, args.beta, min_delta)
    for s in pool.starmap(Sample, map(f, range(start, stop))):
        row = [ g(s) for g in (int, bool, float) ]
        writer.writerow(dict(zip(fieldnames, row)))
