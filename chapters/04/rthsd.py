import sys
import csv
from argparse import ArgumentParser

from irstats import Scores, RandomisedTukey

arguments = ArgumentParser()
arguments.add_argument('--baseline')
arguments.add_argument('--B', type=int, default=1)
arguments.add_argument('--workers', type=int)
args = arguments.parse_args()

assert(args.B > 0)

writer = None
scores = Scores.from_csv(sys.stdin)
rthsd = RandomisedTukey(scores, args.B, args.workers, args.baseline)

for i in rthsd:
    if writer is None:
        writer = csv.DictWriter(sys.stdout, fieldnames=i._fields)
        writer.writeheader()
    writer.writerow(i._asdict())
