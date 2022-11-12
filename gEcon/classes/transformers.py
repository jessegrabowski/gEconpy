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
        return np.exp(x)

    def unconstrain(self, x):
        return np.log(x)

    def jac_det(self, x):
        return np.exp(x)

    def sp_jac_det(self, x):
        return np.exp(x)


class IntervalTransformer(Transformer):

    def __init__(self, low=0, high=1):
        self.low = low
        self.high = high

    def constrain(self, x):
        sigmoid_x = expit(x)
        return sigmoid_x * self.high + (1 - sigmoid_x) * self.low

    def unconstrain(self, x):
        return np.log(x - self.low) - np.log(self.high - x)

    def jac_det(self, x):
        return (self.high - self.low) * np.exp(-x) / (1 + np.exp(-x)) ** 2

    def sp_jac_def(self, x):
        return (self.high - self.low) * sp.exp(-x) / (1 + sp.exp(-x)) ** 2
