from scipy import stats
from statsmodels.stats import libqsturng

__all__ = [
    'Inverse',
    'TInverse',
    'QInverse',
]

class Inverse:
    def __init__(self, alpha, df, tail=True):
        self.alpha = alpha
        self.df = df

    def __float__(self):
        return float(self.inv())

    def inv(self):
        raise NotImplementedError()

#
# https://support.office.com/en-us/article/t-inv-2t-function-ce72ea19-ec6c-4be7-bed2-b9baf2264f17
# EXCEL::T.INV.2T(p, df) == t.ppf(1 - (p / 2), df)
#
class TInverse(Inverse):
    def __init__(self, alpha, df, tail=True):
        super().__init__(alpha, df)

        self.p = 1 - self.alpha
        if tail:
            self.p /= 2

    def inv(self):
        return stats.t.ppf(1 - self.p, self.df)

class QInverse(Inverse):
    def __init__(self, alpha, df, systems):
        super().__init__(alpha, df)
        self.systems = systems

    def inv(self):
        return libqsturng.qsturng(self.alpha, self.systems, self.df)
