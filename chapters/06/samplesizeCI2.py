import sys
import csv
import math
import logging
import operator as op
import itertools as it
import functools as ft
import scipy.stats as st
import multiprocessing as mp
from argparse import ArgumentParser

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S')

# def valid(args):
#     focus = ('effect_size', 'difference', 'sigma')
#     (a, *b) = [ bool(getattr(args, x)) for x in focus ]

#     return a ^ op.or_(*b)

class Sample:
    def __init__(self, n, alpha, delta, sigma):
        self.n = n
        phi = self.n - 1
        t = st.t.ppf(1 - alpha / 2, phi)

        self.left = delta / (2 * math.sqrt(sigma))
        self.right = t * (1 - 1 / (4 * phi)) / math.sqrt(self.n)

    def __iter__(self):
        yield from (self.left, self.right)

    def __int__(self):
        return self.n

    def __bool__(self):
        return bool(self.right <= self.left)

arguments = ArgumentParser()
arguments.add_argument('--alpha', type=float,
                       help='Type I Error probability')
arguments.add_argument('--delta', type=float,
                       help='Cap on the expected width of the CI')
arguments.add_argument('--sigma', type=float,
                       help='Variance estimate for score differences')
arguments.add_argument('--radius', type=int, default=0)
arguments.add_argument('--workers', type=int, default=mp.cpu_count())
args = arguments.parse_args()

# if not valid(args):
#     raise ValueError('Must specify either an effect '
#                      'size, or a difference and sigma, '
#                      'but not both.')

with mp.Pool(args.workers) as pool:
    zalpha = st.norm.ppf(1 - args.alpha / 2)
    n = 4 * zalpha ** 2 * args.sigma / args.delta ** 2

    logging.debug(n)

    fieldnames = ('n', 'adequate', 'left', 'right')
    writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames)
    writer.writeheader()

    f = ft.partial(Sample,
                   alpha=args.alpha,
                   delta=args.delta,
                   sigma=args.sigma)
    for s in pool.imap_unordered(f, it.count(math.floor(n))):
        found = bool(s)
        row = [
            int(s),
            found,
            *list(s),
        ]
        writer.writerow(dict(zip(fieldnames, row)))
        if found:
            break
