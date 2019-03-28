import sys
import csv
from argparse import ArgumentParser

from irstats import Scores, Welch

arguments = ArgumentParser()
arguments.add_argument('--alpha', type=float, default=0.95)
args = arguments.parse_args()

assert(0 <= args.alpha <= 1)

results = Scores.from_csv(sys.stdin)
t = Welch(results, args.alpha)

writer = csv.DictWriter(sys.stdout, fieldnames=t.fieldnames)
writer.writeheader()
writer.writerows(t)
