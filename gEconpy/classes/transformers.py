from abc import ABC

import numpy as np
import sympy as sp
from scipy.special import expit


def softplus(x):
    return np.log(1 + np.exp(x))


class Transformer(ABC):
    def constrain(self, x):
        raise NotImplementedError

    def unconstrain(self, x):
        raise NotImplementedError

    def jac_det(self, x):
        raise NotImplementedError

    def sp_jac_det(self, x):
        raise NotImplementedError


class IdentityTransformer(Transformer):
    def constrain(self, x):
        return x

    def unconstrain(self, x):
        return x

    def jac_det(self, x):
        return 1

    def sp_jac_det(self, x):
        return 1


class PositiveTransformer(Transformer):
    def constrain(self, x):
        return x**2

    def unconstrain(self, x):
        return x**0.5

    def jac_det(self, x):
        return 2 * x

    def sp_jac_det(self, x):
        return 2 * x


class IntervalTransformer(Transformer):
    def __init__(self, low=0, high=1, slope=1):
        self.low = low
        self.high = high
        self.slope = slope
        self.eps = 1e-8

    def constrain(self, x):
        sigmoid_x = expit(self.slope * x)
        return sigmoid_x * self.high + (1 - sigmoid_x) * self.low
        # low, high, k, eps = self.low, self.high, self.slope, self.eps
        # return low + (high - low) / (1 + np.exp(-k * x))

    def unconstrain(self, x):
        low, high, k, eps = self.low, self.high, self.slope, self.eps
        return np.log((x - low + eps) / (high - x + eps)) / k

        # return np.log(x - self.low) - np.log(self.high - x)

    def jac_det(self, x):
        return (self.high - self.low) * np.exp(-x) / (1 + np.exp(-x)) ** 2

    def sp_jac_def(self, x):
        return (self.high - self.low) * sp.exp(-x) / (1 + sp.exp(-x)) ** 2
