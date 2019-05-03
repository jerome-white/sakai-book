import sys
import csv
import math
import operator as op
import collections as cl
from argparse import ArgumentParser

from statsmodels.stats.power import FTestAnovaPower

Result = cl.namedtuple('Result', 'fhat, corrected, power, cell, total')

def pwr_anova_pre(f, df, cell, alpha, power):
    df += 1
    n = fpow.solve_power(effect_size=f, k_groups=df, alpha=alpha, power=power)
    result = math.ceil((n - 1) * df / cell + 1)

    return result

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

row = []

fpow = FTestAnovaPower()
ncells = args.m * args.n
phiE = args.n_total - ncells

phiA = args.m - 1
eta2A = phiA * args.fstat_A / (phiA * args.fstat_A + phiE)
fhatA = math.sqrt(eta2A / (1 - eta2A))
numA = phiE / (phiA + 1) + 1
power = fpow.solve_power(effect_size=fhatA,
                         k_groups=args.m,
                         nobs=numA,
                         alpha=args.alpha)
percelsiz_A = pwr_anova_pre(fhatA, phiA, ncells, args.alpha, args.power)
totalsiz_A = ncells * percelsiz_A

row.append(Result(fhatA, numA, power, percelsiz_A, totalsiz_A))

phiB = args.n - 1
eta2B = phiB * args.fstat_B / (phiB * args.fstat_B + phiE)
fhatB = math.sqrt(eta2B / (1 - eta2B))
numB = phiE / (phiB + 1) + 1
power = fpow.solve_power(effect_size=fhatB,
                         k_groups=args.n,
                         nobs=numB,
                         alpha=args.alpha)
percelsiz_B = pwr_anova_pre(fhatB, phiB, ncells, args.alpha, args.power)
totalsiz_B = ncells * percelsiz_B

row.append(Result(fhatB, numB, power, percelsiz_B, totalsiz_B))

if args.fstat_AB is not None:
    phiAB = (args.m - 1) * (args.n - 1)
    eta2AB = phiAB * args.fstat_AB / (phiAB * args.fstat_AB + phiE)
    fhatAB = math.sqrt(eta2AB / (1 - eta2AB))
    numAB = phiE / (phiAB + 1) + 1
    power = fpow.solve_power(effect_size=fhatAB,
                             k_groups=phiAB + 1,
                             nobs=numAB,
                             alpha=args.alpha)
    percelsiz_AB = pwr_anova_pre(fhatAB,
                                 phiAB,
                                 ncells,
                                 args.alpha,
                                 args.power)
    totalsiz_AB = ncells * percelsiz_AB
    row.append(Result(fhatAB, numAB, power, percelsiz_AB, totalsiz_AB))

writer = csv.DictWriter(sys.stdout, fieldnames=Result._fields)
writer.writeheader()
writer.writerows(map(op.methodcaller('_asdict'), row))
