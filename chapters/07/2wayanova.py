import sys
import csv
import math
import functools as ft
from argparse import ArgumentParser

from statsmodels.stats.power import FTestAnovaPower

class AnovaPowerTwo:
    def __init__(self, fstat, m, n, total, alpha, phi):
        self.m = m
        self.phi = phi
        self.ncells = m * n

        phiE = total - self.ncells
        eta2 = self.phi * fstat / (self.phi * fstat + phiE)
        self.fhat = math.sqrt(eta2 / (1 - eta2))
        self.num = phiE / (phi + 1) + 1

        self.pwr = ft.partial(FTestAnovaPower().solve_power,
                              effect_size=self.fhat,
                              alpha=alpha)

    def __float__(self):
        return float(self.pwr(k_groups=self.groups, nobs=self.num))

    def pre(self, power):
        df = self.phi + 1
        n = self.pwr(k_groups=df, power=power)

        percelsiz = math.ceil((n - 1) * df / self.ncells + 1)
        totalsiz = self.ncells * percelsiz

        return (percelsiz, totalsiz)

    @property
    def groups(self):
        raise NotImplementedError()

class Factor(AnovaPowerTwo):
    def __init__(self, fstat, m, n, total, alpha):
        super().__init__(fstat, m, n, total, alpha, m - 1)

    @property
    def groups(self):
        return self.m

class Interaction(AnovaPowerTwo):
    def __init__(self, fstat, m, n, total, alpha):
        super().__init__(fstat, m, n, total, alpha, (m - 1) * (n - 1))

    @property
    def groups(self):
        return self.phi + 1

arguments = ArgumentParser()
arguments.add_argument('--fstat-A', type=float,
                       help='F-statistic (F0) reported')
arguments.add_argument('--fstat-B', type=float,
                       help='F-statistic (F0) reported')
arguments.add_argument('--fstat-AB', type=float,
                       help='F-statistic (F0) reported')
arguments.add_argument('--m', type=int,
                       help='Number of groups')
arguments.add_argument('--n', type=int,
                       help='Sample size')
arguments.add_argument('--n-total', type=int,
                       help='Sample size')
arguments.add_argument('--alpha', type=float, default=0.05,
                       help='Type I Error probability')
arguments.add_argument('--power', type=float, default=0.80,
                       help='Desired statistical power')
args = arguments.parse_args()

params = [
    (args.fstat_A, Factor, lambda x: x),
    (args.fstat_B, Factor, lambda x: reversed(x)),
    (args.fstat_AB, Interaction, lambda x: x),
]
dimensions = (args.m, args.n)

fieldnames = [
    'fhat',
    'corrected',
    'power',
    'cell',
    'total',
]
writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames)
writer.writeheader()

for (fstat, AnovaPower, f) in params:
    if fstat is not None:
        apwr = AnovaPower(fstat, *f(dimensions), args.n_total, args.alpha)
        fieldvalues = [
            apwr.fhat,
            apwr.num,
            float(apwr),
            *apwr.pre(args.power),
        ]

        writer.writerow(dict(zip(fieldnames, fieldvalues)))
