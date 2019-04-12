import sys
import csv
import logging
from argparse import ArgumentParser

from irstats import Scores, OneWay, TwoWay, Tukey

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S')

arguments = ArgumentParser()
arguments.add_argument('--alpha', type=float, default=0.95)
args = arguments.parse_args()

assert(0 <= args.alpha <= 1)

scores = Scores.from_csv(sys.stdin)

for i in (OneWay, TwoWay):
    logging.debug(i)
    tukey = Tukey(scores, args.alpha, i)

    writer = None
    for i in tukey:
        if writer is None:
            writer = csv.DictWriter(sys.stdout, fieldnames=i.keys())
            writer.writeheader()
        writer.writerow(i)
