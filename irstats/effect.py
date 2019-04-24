import math

import numpy as np

class Effect:
    def __init__(self, x1, x2):
        self.x = (x1, x2)
        self.difference = np.subtract(*map(np.mean, self.x))
        (self.s, self.n) = map(sum, [ map(x, self.x) for x in (self.S, len) ])

    def __float__(self):
        return self.correction(self.difference / math.sqrt(self.V()))

    def S(self, values):
        return np.sum(np.square(np.subtract(values, np.mean(values))))

    def V(self):
        raise NotImplementedError()

class Hedge(Effect):
    def __init__(self, x1, x2, unbiased=False):
        super().__init__(x1, x2)

        # Equation 5.10
        if unbiased:
            self.correction = lambda x: (1 - 3 / (4 * self.n - 9)) * x

    # Equation 5.6
    def V(self):
        return self.s / (self.n - 2)

class Cohen(Effect):
    # Equation 5.8
    def V(self):
        return self.s / self.n

class Glass(Effect):
    def __init__(self, x1, x2, baseline=None):
        super().__init__(x1, x2)

        if baseline is None:
            self.baseline = x2
        else:
            self.baseline = baseline

            # Equation 5.12
            n2 = len(self.baseline)
            self.correction = lambda x: (1 - 3 / (4 * n2 - 5)) * x

    # Equation 5.11
    def V(self):
        return self.S(self.baseline) / (len(self.baseline) - 1)

class TukeyEffect(Effect):
    def __init__(self, x1, x2, anova):
        super().__init__(x1, x2)
        self.anova = anova

    def V(self):
        return self.anova.phiE
