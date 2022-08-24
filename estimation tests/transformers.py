import numpy as np


class IdentityTransformer:
    def __init__(self):
        pass

    def param_space_to_real_line(self, x):
        return x

    def real_line_to_param_space(self, x):
        return x


class RangeTransformer:

    def __init__(self, x_min=0, x_max=1, slope=1):
        self.x_min = x_min
        self.x_max = x_max
        self.slope = slope

        self.eps = np.spacing(1)

    def param_space_to_real_line(self, x):
        x_min, x_max, k, eps = self.x_min, self.x_max, self.slope, self.eps
        return np.log((x - x_min + eps) / (x_max - x + eps)) / k

    def real_line_to_param_space(self, x):
        x_min, x_max, k, eps = self.x_min, self.x_max, self.slope, self.eps
        return x_min + (x_max - x_min) / (1 + np.exp(-k * x))

class PositiveTransformer:

    def __init__(self):
        self.sign = 1

    def param_space_to_real_line(self, x):
        return np.sqrt(x) * self.sign

    def real_line_to_param_space(self, x):
        self.sign = np.sign(x)
        return x ** 2
