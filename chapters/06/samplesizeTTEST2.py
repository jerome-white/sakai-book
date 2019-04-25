import math
import logging
import operator as op
import scipy.stats as st
import multiprocessing as mp
from argparse import ArgumentParser

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S')

def func(incoming, outgoing, args):
    delta = args.min_diff / math.sqrt(args.sigma)

    while True:
        n = incoming.get()
        lamb = math.sqrt(n) * delta
        phi = n - 1

        logging.debug('{} lambda:{} phi:{}'.format(n, lamb, phi))

        w = st.t.ppf(1 - args.alpha / 2, phi)

        normalization = math.sqrt(1 + w ** 2 / (2 * phi))
        f = lambda x: (x * (1 - 1 / (4 * phi)) - lamb) / normalization
        (left, right) = [ st.norm.cdf(f(g(w))) for g in (op.neg, op.pos) ]

        beta = left + 1 - right

        logging.debug('{} {}: {} {}'.format(n, w, left, right))

        outgoing.put((n, beta))

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

incoming = mp.Queue()
outgoing = mp.Queue()
with mp.Pool(args.workers, func, (outgoing, incoming, args)):
    norminv = lambda x: st.norm.ppf(x)
    (zalpha, zbeta) = map(norminv, (1 - args.alpha / 2, args.beta))

    min_delta = args.min_diff / math.sqrt(args.sigma)
    n = ((zalpha - zbeta) / min_delta) ** 2 + zalpha ** 2 / 2
    logging.debug(n)

    recommended = math.ceil(n)
    start = max(0, recommended - args.radius)
    stop = recommended + args.radius + 1
    for i in range(start, stop):
        outgoing.put(i)

    for _ in range(stop - start):
        (n, beta) = incoming.get()
        logging.info('{} {}'.format(n, 1 - args.beta < beta))
