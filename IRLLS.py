import numpy as np
from enum import Enum


class WeightingType(Enum):
    HUBER = 1
    BISQUARE = 2
    CAUCHY = 3
    FAIR = 4
    WELCH = 5
    OLLS = 6


def WeightedOLLS(A, w, b):
    sw = np.sqrt(w)
    W = np.diag(np.squeeze(sw))
    AW = np.dot(W, A)
    bW = b * sw
    xW = np.linalg.lstsq(AW, bW)[0]
    Residuals = np.abs(A @ xW - b)
    return xW, Residuals


class IRLLS:

    def __init__(self, A, b, method: WeightingType = WeightingType.OLLS, params=None, eps_x=1e-6, max_iter=100):
        self.A = A
        self.b = b
        self.w = np.ones_like(b)
        self.x, self.Ar = WeightedOLLS(A, self.w, b)
        self.eps_x = eps_x
        self.max_iter = max_iter
        self.method = method
        if isinstance(params, list):
            self.params = params
        else:
            self.params = [1.0]
        self.dx = np.inf

    def SetDefaultParams(self):
        if self.method == WeightingType.HUBER:
            self.params[0] = 1.345
        if self.method == WeightingType.BISQUARE:
            self.params[0] = 4.685
        if self.method == WeightingType.FAIR:
            self.params[0] = 1.400
        if self.method == WeightingType.CAUCHY:
            self.params[0] = 2.385
        if self.method == WeightingType.WELCH:
            self.params[0] = 2.985

    def GetWeights(self):
        if self.method == WeightingType.HUBER:
            e = self.Ar / np.median(self.Ar) * 0.6745 / self.params[0]
            self.w = 1.0 / (e + 1e-3)
            self.w[self.Ar < 1] = 1
        if self.method == WeightingType.BISQUARE:
            e = self.Ar / np.median(self.Ar) * 0.6745 / self.params[0]
            self.w = (1.0 + e**2)**2
            self.w[e > 1] = 0
        if self.method == WeightingType.FAIR:
            e = self.Ar / np.median(self.Ar) * 0.6745 / self.params[0]
            self.w = 1.0 / (e + 1.0)
        if self.method == WeightingType.CAUCHY:
            e = self.Ar / np.median(self.Ar) * 0.6745 / self.params[0]
            self.w = 1.0 / (e**2 + 1.0)
        if self.method == WeightingType.WELCH:
            e = self.Ar / np.median(self.Ar) * 0.6745 / self.params[0]
            self.w = np.exp(-e**2)

    def Iterate(self):
        c_iter = 1
        if self.method == WeightingType.OLLS:
            return
        while self.dx > self.eps_x and (c_iter < self.max_iter):
            px = np.copy(self.x)
            self.GetWeights()
            self.x, self.Ar = WeightedOLLS(self.A, self.w, self.b)
            self.dx = np.linalg.norm(self.x - px)
        return self.x
