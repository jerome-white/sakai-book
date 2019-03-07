import sys
import csv
import itertools as it
import collections as cl
import multiprocessing as mp
from pathlib import Path
from argparse import ArgumentParser

import numpy as np

from irstats.systems import Systems

class RandomisedSystems(Systems):
    def shuffle(self):
        return np.apply_along_axis(np.random.permutation, 1, self.data)

def func(incoming, outgoing, systems):
    d = dict(systems.differences())

    while True:
        _ = incoming.get()

        x = systems.shuffle().mean(axis=0)
        d_ = x.max() - x.min()
        for j in systems.pairs():
            if d_ >= abs(d[j]):
                outgoing.put(j)

        outgoing.put(None)

arguments = ArgumentParser()
arguments.add_argument('--B', type=int, default=0)
arguments.add_argument('--workers', type=int)
args = arguments.parse_args()

assert(args.B > 0)

incoming = mp.JoinableQueue()
outgoing = mp.Queue()
systems = RandomisedSystems(sys.stdin)

with mp.Pool(args.workers, func, (outgoing, incoming, systems)):
    for i in range(args.B):
        outgoing.put(None)

    jobs = args.B
    count = cl.Counter()
    while jobs:
        pair = incoming.get()
        if pair is None:
            jobs -= 1
        else:
            count[pair] += 1

fieldnames = [
    'system_1',
    'system_2',
    'difference',
    'p-value',
]
writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames)
writer.writeheader()
for (i, j) in systems.differences():
    row = (*i, j, count[i] / args.B)
    writer.writerow(dict(zip(fieldnames, row)))
