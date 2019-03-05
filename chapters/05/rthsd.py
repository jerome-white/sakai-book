import itertools as it
import collections as cl
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

arguments = ArgumentParser()
arguments.add_argument('--B', type=int, default=0)
arguments.add_argument('--systems', type=Path)
args = arguments.parse_args()

assert(args.B > 0)

systems = Systems(args.systems)

d = {}
for i in systems.pairs():
    d[i] = np.subtract(*[ systems[x].mean() for x in i ])

count = cl.Counter()
for i in range(args.B):
    U = systems.shuffle()
    x = U.mean(axis=0)
    d_ = x.max() - x.min()
    for j in systems.pairs():
        if d_ >= abs(d[j]):
            count[j] += 1

p_value = { x: count[x] / args.B for x in systems.pairs() }

for (i, j) in p_value.items():
    print(*i, ':', j)
