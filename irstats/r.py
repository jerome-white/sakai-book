import itertools as it
import collections as cl

import rpy2.robjects.packages as rpk
from rpy2.robjects.vectors import DataFrame

Column = cl.namedtuple('Column', 'no, name')

class R:
    def __init__(self, df):
        self.df = df
        self.stats = rpk.importr('stats')

    def __iter__(self):
        for i in it.combinations(enumerate(self.df.colnames), r=2):
            yield tuple(it.starmap(Column, i))

    def ttest(self, *args, **kwargs):
        fields = {
            'statistic': 't',
            'parameter': 'df',
            'p.value': 'p-value',
            'conf.int': ('left_ci', 'right_ci'),
        }
        if 'paired' in kwargs and kwargs['paired']:
            fields['estimate'] = 'difference'

        for i in self:
            systems = [ self.df.rx2(x.name) for x in i ]
            t = self.stats.t_test(*systems, *args, **kwargs)

            result = {}
            for (k, v) in fields.items():
                values = list(t.rx2(k))
                if isinstance(v, (list, tuple)):
                    for (x, y) in zip(v, values):
                        result[x] = y
                else:
                    result[v] = values.pop()

            yield result

    @classmethod
    def from_csv(cls, data):
        return cls(DataFrame.from_csvfile(str(data)))
