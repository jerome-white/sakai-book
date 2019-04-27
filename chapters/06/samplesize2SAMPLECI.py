import sys
import csv
import math
import logging
import itertools as it
import functools as ft
from argparse import ArgumentParser

import scipy.stats as st

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S')

class Sample:
    def __init__(self, n, alpha, delta, sigma):
        self.n = n
        self.phi = 2 * self.n - 2
        self.t = st.t.ppf(1 - alpha / 2, self.phi)

    def __iter__(self):
        yield from (self.left, self.right)

    def __int__(self):
        return self.n

    def __bool__(self):
        return bool(self.right <= self.left)

class Approximation(Sample):
    def __init__(self, n, alpha, delta, sigma):
        super().__init__(n, alpha, delta, sigma)

        self.left = delta / (2 * math.sqrt(2) * math.sqrt(sigma))
        self.right = self.t * (1 - 1 / (4 * self.phi)) / math.sqrt(self.n)

class WithGamma(Sample):
    def __init__(self, n, alpha, delta, sigma):
        super().__init__(n, alpha, delta, sigma)

        g1 = math.gamma((self.phi + 1) / 2)
        g2 = math.gamma(self.phi / 2)

        self.left = delta / (4 * math.sqrt(sigma))
        self.right = self.t * g1 / (math.sqrt(self.n * self.phi) * g2)

arguments = ArgumentParser()
arguments.add_argument('--alpha', type=float,
                       help='Type I Error probability')
arguments.add_argument('--delta', type=float,
                       help='Cap on the expected width of the CI')
arguments.add_argument('--sigma', type=float,
                       help='Variance estimate for score differences')
arguments.add_argument('--use-gamma', action='store_true')
args = arguments.parse_args()

S = WithGamma if args.use_gamma else Approximation
zalpha = st.norm.ppf(1 - args.alpha / 2)
n = 8 * zalpha ** 2 * args.sigma / args.delta ** 2

logging.debug(n)

fieldnames = ('n', 'adequate', 'left', 'right')
writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames)
writer.writeheader()

for i in it.count(math.floor(n)):
    s = S(i, args.alpha, args.delta, args.sigma)

    found = bool(s)
    row = [
        int(s),
        found,
        *list(s),
    ]
    writer.writerow(dict(zip(fieldnames, row)))

    if found:
        break
