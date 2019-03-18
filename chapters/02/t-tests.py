import sys
import csv
import functools as ft
from argparse import ArgumentParser

import scipy.stats as st

import irstats as irs

class TTest:
    def __init__(self, x, y, name):
        self.x = x
        self.y = y
        self.name = name
        
    def __str__(self):
        return self.name

    def __call__(self):
        return self.test(self.x, self.y)
    
class Paired(TTest):
    def	__init__(self, x, y):
        super().__init__(x, y, 'paired')
        self.test = st.ttest_rel

class Student(TTest):
    def	__init__(self, x, y):
        super().__init__(x, y, 'student')
        self.test = ft.partial(st.ttest_ind, equal_var=True)

class Welch(TTest):
    def	__init__(self, x, y):
        super().__init__(x, y, 'Welch')
        self.test = ft.partial(st.ttest_ind, equal_var=False)

arguments = ArgumentParser()
arguments.add_argument('--alpha', type=float, default=0.95)
args = arguments.parse_args()

assert(0 <= args.alpha <= 1)

systems = irs.Systems(sys.stdin)

fieldnames = [
    'test',    
    'system_1',
    'system_2',
    'difference',
    't-statistic',
    'p-value',
]
writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames)
writer.writeheader()

for T in (Paired, Student, Welch):
    for (i, j) in systems.differences():
        tstat = T(*[ systems[x] for x in i ])
        row = [
            str(tstat),
            *i,
            j,
            *tstat(),
        ]
        writer.writerow(dict(zip(fieldnames, row)))
