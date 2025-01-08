from enum import Enum
import numpy as np


class OutlierType(Enum):
    UNIFORM = 0
    CONSTANT = 1


class Line:

    OUTLIERS = OutlierType.UNIFORM

    def __init__(self, n: int, x_range: list, epsilon: float, no: int = 0):
        self.n = n
        self.no = no
        self.x_range = x_range
        self.np = epsilon
        self.no = no

    def Generate(self, a: float, b: float):
        x0 = self.x_range[0] + np.random.rand(1, self.n) * (self.x_range[1] - self.x_range[0])
        x0 = np.sort(x0)
        y0 = a * x0 + b
        yn = y0 + np.random.randn(1, self.n) * self.np
        y_max = np.max(y0)
        y_min = np.min(y0)
        if self.no > 0:
            Indices = np.random.permutation(self.n)
            Indices = Indices[0:self.no]
            if Line.OUTLIERS == OutlierType.UNIFORM:
                c_outliers = y_min + np.random.rand(1, self.no) * (y_max - y_min)
            else:
                c_outliers = y_min + np.random.rand(1) * (y_max - y_min)
            yn[0, Indices] = c_outliers
        return x0, y0, yn
