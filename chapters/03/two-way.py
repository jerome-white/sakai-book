import sys
import csv
import logging
from argparse import ArgumentParser

from irstats import Scores, TwoWay

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S')

arguments = ArgumentParser()
arguments.add_argument('--alpha', type=float, default=0.95)
args = arguments.parse_args()

assert(0 <= args.alpha <= 1)

scores = Scores.from_csv(sys.stdin)
anova = TwoWay(scores, args.alpha)

writer = None
for i in anova:
    if writer is None:
        writer = csv.DictWriter(sys.stdout, fieldnames=i._fields)
        writer.writeheader()
    writer.writerow(i._asdict())

for (i, j) in anova.ci():
    logging.info('{name} {ci}'.format(name=i, ci=list(j)))
