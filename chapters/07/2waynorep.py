import sys
import csv
import math
from argparse import ArgumentParser

from statsmodels.stats.power import FTestPower

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


n = args.n - 1
m = args.m - 1
v = n * m

# FHAT2 <- FSTAT/(N-1)
es = args.fstat / n

kwargs = {
    'effect_size': es,
    'alpha': args.alpha,
    'df_num': m,
}

fpow = FTestPower()

#ACPOWER <- pwr.f2.test( u=M-1, v=(M-1)*(N-1), FHAT2, sig.level=ALPHA )$power
power = fpow.solve_power(df_denom=v, **kwargs)

# SAMPLESIZ <- ceiling( ( pwr.f2.test(u=M-1, power=POW, f2=FHAT2, sig.level=ALPHA )$v + N-1 + M-1 + 1)/M )
sample = (fpow.solve_power(power=args.power, **kwargs) + n + m + 1) / args.m
sample = math.ceil(sample)

row = [
    es,
    power,
    sample,
]

fieldnames = ('fhat', 'power', 'size')
writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames)
writer.writeheader()
writer.writerow(dict(zip(fieldnames, row)))
