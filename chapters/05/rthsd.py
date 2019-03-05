import sys
import itertools as it
import collections as cl
import multiprocessing as mp
from pathlib import Path
from argparse import ArgumentParser

import numpy as np

class Systems:
    def __init__(self, data):
        self.data = np.loadtxt(data, delimiter=',', skiprows=1)
        (self.runs, self.systems) = self.data.shape

    def __getitem__(self, key):
        return self.data[:,key]

    def pairs(self):
        yield from it.combinations(range(self.systems), r=2)

    def shuffle(self):
        return np.apply_along_axis(np.random.permutation, 1, self.data)

def func(incoming, outgoing, data):
    systems = Systems(args.systems)

    d = {}
    for i in systems.pairs():
        d[i] = np.subtract(*[ systems[x].mean() for x in i ])

    while True:
        _ = incoming.get()

        U = systems.shuffle()
        x = U.mean(axis=0)
        d_ = x.max() - x.min()
        for j in systems.pairs():
            if d_ >= abs(d[j]):
                outgoing.put(j)

        outgoing.put(None)

arguments = ArgumentParser()
arguments.add_argument('--B', type=int, default=0)
arguments.add_argument('--systems', type=Path)
arguments.add_argument('--workers', type=int)
args = arguments.parse_args()

assert(args.B > 0)

incoming = mp.JoinableQueue()
outgoing = mp.Queue()

with mp.Pool(args.workers, func, (outgoing, incoming, args.systems)):
    for i in range(args.B):
        outgoing.put(i)

    jobs = args.B
    count = cl.Counter()
    while jobs:
        pair = incoming.get()
        if pair is None:
            jobs -= 1
        else:
            count[pair] += 1

systems = Systems(args.systems)
p_value = { x: count[x] / args.B for x in systems.pairs() }

for (i, j) in p_value.items():
    print(*i, ':', j)
