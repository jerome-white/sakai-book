import math

import numpy as np

class Effect:
    def __init__(self, x1, x2):
        self.x = (x1, x2)
        self.difference = np.subtract(*map(np.mean, self.x))
        (self.s, self.n) = map(sum, [ map(x, self.x) for x in (self.S, len) ])

    def __float__(self):
        return self.difference / math.sqrt(self.V())

    def S(self, values):
        return np.sum(np.square(np.subtract(values, np.mean(values))))

    def V(self):
        raise NotImplementedError()

class Cohen(Effect):
    # Equation 5.8
    def V(self):
        return self.s / self.n

class TukeyEffect(Effect):
    def __init__(self, x1, x2, anova):
        super().__init__(x1, x2)
        self.anova = anova

    def V(self):
        return self.anova.phiE

class UnbiasedEffect(Effect):
    def __init__(self, x1, x2, desired=False):
        super().__init__(x1, x2)
        self.desired = desired

    def __float__(self):
        value = super().__float__()
        if self.desired:
            value *= self.unbias()

        return value

    def unbias(self):
        raise NotImplementedError()

class Hedge(UnbiasedEffect):
    # Equation 5.6
    def V(self):
        return self.s / (self.n - 2)

    # Equation 5.10
    def unbias(self):
        return 1 - 3 / (4 * self.n - 9)

class Glass(UnbiasedEffect):
    def __init__(self, x1, x2, baseline=None):
        super().__init__(x1, x2, bool(baseline))
        self.baseline = x2 if baseline is None else baseline

    # Equation 5.11
    def V(self):
        return self.S(self.baseline) / (len(self.baseline) - 1)

    # Equation 5.12
    def unbias(self):
        return 1 - 3 / (4 * len(self.baseline) - 5)
