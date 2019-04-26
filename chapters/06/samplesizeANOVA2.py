import sys
import csv
import math
import logging
import scipy.stats as st
import multiprocessing as mp
from pathlib import Path
from argparse import ArgumentParser

import pandas as pd

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S')

class Sample:
    def __init__(self, n, alpha, beta, delta, m):
        self.n = n
        self.beta = beta

        phiA = m - 1
        phiE = m * (self.n - 1)
        w = st.f.ppf(1 - alpha, phiA, phiE)
        cA = (phiA + 2 * self.n * delta) / (phiA + self.n * delta)
        phiA_ = (phiA + self.n * delta) ** 2 / (phiA + 2 * self.n * delta)
        self.u = ((math.sqrt(w / phiE) *
                   math.sqrt(2 * phiE - 1) -
                   math.sqrt(cA / phiA) *
                   math.sqrt(2 * phiA_ - 1)) /
                  math.sqrt(cA / phiA - w / phiE))

    def __float__(self):
        return float(1 - st.norm.cdf(self.u))

    def __int__(self):
        return self.n

    def __bool__(self):
        return 1 - self.beta < float(self)

class Parameters(dict):
    def __str__(self):
        return ' '.join([ x + ': ' + str(y) for (x, y) in self.items() ])

def func(incoming, outgoing, args):
    while True:
        params = incoming.get()

        lamb = params['x1'] + params['x2'] * math.sqrt(args.systems - 1)
        delta = args.difference ** 2 / (2 * args.sigma)
        n = lamb / delta

        logging.debug('n: {} {}'.format(n, params))

        for i in range(math.floor(n), 2, -1):
            s = Sample(i, params['alpha'], params['beta'], delta, args.systems)
            outgoing.put((params, s))
            if not s:
                break

arguments = ArgumentParser()
arguments.add_argument('--difference', type=float,
                       help='Minimum detectable difference')
arguments.add_argument('--sigma', type=float,
                       help='Variance estimate for score differences')
arguments.add_argument('--systems', type=int,
                       help='Number of systems compared')
arguments.add_argument('--approximations', type=Path,
                       help='Table 6.4')
arguments.add_argument('--workers', type=int, default=mp.cpu_count())
args = arguments.parse_args()

incoming = mp.Queue()
outgoing = mp.Queue()

with mp.Pool(args.workers, func, (outgoing, incoming, args)):
    df = pd.read_csv(args.approximations, index_col=None)
    for i in df.itertuples(index=False):
        outgoing.put(Parameters(i._asdict()))

    keys = ('n', 'adequate', 'achieved')
    writer = None
    jobs = len(df)

    while jobs:
        (params, sample) = incoming.get()

        values = [ f(sample) for f in (int, bool, float) ]
        row = { x: params[x] for x in ('alpha', 'beta') }
        row.update(dict(zip(keys, values)))

        if writer is None:
            writer = csv.DictWriter(sys.stdout, fieldnames=row.keys())
            writer.writeheader()
        writer.writerow(row)

        if not sample:
            jobs -= 1
