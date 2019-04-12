import sys
import csv
from argparse import ArgumentParser

from irstats import Scores, RandomisedTukey

arguments = ArgumentParser()
arguments.add_argument('--B', type=int, default=0)
arguments.add_argument('--workers', type=int)
args = arguments.parse_args()

assert(args.B > 0)

scores = Scores.from_csv(sys.stdin)
writer = None

for i in RandomisedTukey(scores, args.B, args.workers):
    if writer is None:
        writer = csv.DictWriter(sys.stdout, fieldnames=i._fields)
        writer.writeheader()
    writer.writerow(i._asdict())
