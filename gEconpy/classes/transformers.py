from abc import ABC

import numpy as np
from scipy.special import expit


class Transformer(ABC):
    def constrain(self, x):
        raise NotImplementedError

    def unconstrain(self, x):
        raise NotImplementedError


class IdentityTransformer(Transformer):
    def constrain(self, x):
        return x

    def unconstrain(self, x):
        return x


class PositiveTransformer(Transformer):
    def __init__(self):
        self.last_sign = 1

    def constrain(self, x):
        self.last_sign = np.sign(x)
        return x**2

    def unconstrain(self, x):
        return x**0.5 * self.last_sign


class IntervalTransformer(Transformer):
    def __init__(self, low=0, high=1, slope=1):
        self.low = low
        self.high = high
        self.slope = slope
        self.eps = 1e-8

    def constrain(self, x):
        sigmoid_x = expit(self.slope * x)
        return sigmoid_x * self.high + (1 - sigmoid_x) * self.low

    def unconstrain(self, x):
        low, high, k, eps = self.low, self.high, self.slope, self.eps
        return np.log((x - low + eps) / (high - x + eps)) / k
