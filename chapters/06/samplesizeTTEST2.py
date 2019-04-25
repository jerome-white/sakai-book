import math
import logging
import operator as op
import scipy.stats as st
import multiprocessing as mp
from argparse import ArgumentParser

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S')

class Sample:
    def __init__(self, n, alpha, beta, delta):
        self.n = n
        self.beta = beta

        lamb = math.sqrt(self.n) * delta
        phi = self.n - 1
        w = st.t.ppf(1 - args.alpha / 2, phi)

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
arguments.add_argument('--min-diff', type=float,
                       help='Minimum detectable difference')
arguments.add_argument('--sigma', type=float,
                       help='Variance estimate for score differences')
arguments.add_argument('--radius', type=int, default=0)
arguments.add_argument('--workers', type=int, default=mp.cpu_count())
args = arguments.parse_args()

with mp.Pool(args.workers) as pool:
    norminv = lambda x: st.norm.ppf(x)
    (zalpha, zbeta) = map(norminv, (1 - args.alpha / 2, args.beta))

    min_delta = args.min_diff / math.sqrt(args.sigma)
    n = ((zalpha - zbeta) / min_delta) ** 2 + zalpha ** 2 / 2

    logging.debug(n)

    recommended = math.ceil(n)
    start = max(2, recommended - args.radius)
    stop = recommended + args.radius + 1

    f = lambda x: (x, args.alpha, args.beta, min_delta)
    for s in pool.starmap(Sample, map(f, range(start, stop))):
        logging.info('{} {}'.format(int(s), bool(s)))
