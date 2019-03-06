import sys
import csv
import itertools as it
import collections as cl
import multiprocessing as mp
from pathlib import Path
from argparse import ArgumentParser

import numpy as np

class Systems:
    def __init__(self, fp):
        cols = fp.readline().rstrip().split(',')
        self.columns = { y: x for (x, y) in enumerate(cols) }
        self.data = np.loadtxt(fp, delimiter=',')
        # (self.runs, self.systems) = self.data.shape

    def __getitem__(self, key):
        index = self.columns[key]
        return self.data[:,index]

    def pairs(self):
        yield from it.combinations(self.columns, r=2)

    def shuffle(self):
        return np.apply_along_axis(np.random.permutation, 1, self.data)

    def differences(self):
        for i in self.pairs():
            yield (i, np.subtract(*[ self[x].mean() for x in i ]))

def func(incoming, outgoing, systems):
    d = dict(systems.differences())

    while True:
        task = incoming.get()

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
systems = Systems(sys.stdin)

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
