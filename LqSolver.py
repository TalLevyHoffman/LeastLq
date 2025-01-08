from numba import njit
from scipy.special import gamma
import numpy as np
import matplotlib.pyplot as plt

from LineGenerator import Line
from LineGenerator import OutlierType


@njit(parallel=True)
def OLLS(x: np.ndarray, y: np.ndarray) -> (float, float):
    x1 = np.sum(x) / x.size
    y1 = np.sum(y) / x.size
    xy = np.sum(x * y) / x.size
    x2 = np.sum(x ** 2) / x.size
    a = (xy - y1 * x1) / (x2 - x1 ** 2)
    b = y1 - a * x1
    return a, b


@njit(parallel=False)
def CalculateResidual(x: np.ndarray, y: np.ndarray, a: float, b: float) -> np.ndarray:
    r = a * x + b - y
    return r


@njit(parallel=False)
def UpdateEstimate(x: np.ndarray, r: np.ndarray, q: float, dq: float) -> (float, float):
    P1 = np.mean(x * np.sign(r) * np.abs(r) ** (q - 1) * np.log(np.abs(r)))
    P2 = np.mean(np.sign(r) * np.abs(r) ** (q - 1) * np.log(np.abs(r)))
    P3 = np.mean(np.abs(r) ** (q - 2))
    P4 = np.mean(x * np.abs(r) ** (q - 2))
    P5 = np.mean(x ** 2. * np.abs(r) ** (q - 2))
    da = dq * (P1 - P4 * P2 / P3) / (P5 - P4 ** 2 / P3) / (q - 1)
    db = (dq * P2 - (q - 1) * da * P4) / P3 / (q - 1)
    return da, db


@njit(parallel=False)
def R2(y: np.ndarray, y_hat: np.ndarray) -> float:
    cc = 1 - np.sum((y - y_hat) ** 2) / np.sum((y - np.mean(y)) ** 2)
    return cc


@njit(parallel=False)
def RMSE(y: np.ndarray, y_hat: np.ndarray) -> float:
    return np.sqrt(np.mean((y - y_hat) ** 2))


@njit(parallel=False)
def Lq(y: np.ndarray, y_hat: np.ndarray, q: float) -> float:
    return np.mean(np.abs(y - y_hat) ** q)


def LqTH(q: float, sigma: float) -> float:
    s = (1 + q) / 2.0
    d = (sigma ** (1 - 2 * s)) * (2.0 ** (1 - s)) * np.sqrt(np.pi)
    n = np.sqrt(2) * gamma(s)
    cLq = n / d
    return cLq


def CheckStop(y: np.ndarray, y_hat: np.ndarray, q: float, sigma: float, fact=1.1, do_filter=False):
    if do_filter:
        res = np.abs(y - y_hat)
        mad = np.median(res)
        y = y[res < (3 * 1.4826 * mad)]
        y_hat = y_hat[res < (3 * 1.4826 * mad)]
    calc_lq = Lq(y, y_hat, q)
    theory_lq = LqTH(q, sigma)
    if calc_lq < (theory_lq * fact):
        return True, calc_lq, theory_lq
    else:
        return False, calc_lq, theory_lq


class LqFitting:

    @staticmethod
    def SolveLq(cx: np.ndarray, cy: np.ndarray, q_final: float, iterations: int = 100, s_mem=False):
        dq = (2.0 - q_final) / float(iterations)
        q = 2 + dq
        a, b = OLLS(cx, cy)
        i_memory = {'a': [], 'b': [], 'q': [], 'Lq': [], 'LTq': []}
        if s_mem:
            i_memory['a'].append(a)
            i_memory['b'].append(b)
            i_memory['q'].append(2)
            i_memory['Lq'].append(np.power(np.sum(np.abs(a*cx + b - cy)**2), 0.5))
        for i in range(iterations):
            q = q - dq
            r = CalculateResidual(cx, cy, a, b)
            da, db = UpdateEstimate(cx, r, q, dq)
            a = a + da
            b = b + db
            if s_mem:
                i_memory['a'].append(a)
                i_memory['b'].append(b)
                i_memory['q'].append(q)
                i_memory['Lq'].append(np.power(np.sum(np.abs(a * cx + b - cy) ** q), 1/q))

        y_hat = a * cx + b
        Statistics = {'Residual': cy - y_hat, 'R2': R2(cy, y_hat), 'RMSE': RMSE(cy, y_hat),
                      'q_final': q_final, 'Lq_final': Lq(cy, y_hat, q), 'Iterations': iterations}

        return a, b, y_hat, Statistics, i_memory

    @staticmethod
    def AutoStopSolveLq(cx: np.ndarray, cy: np.ndarray, c_sigma: float, iterations: int = 100, s_mem=False):
        dq = 1 / float(iterations)
        qF = 1.05
        q = 2 + dq
        q_final = 2.0
        stop = False
        a, b = OLLS(cx, cy)
        i_memory = {'a': [], 'b': [], 'q': [], 'Lq': [], 'LTq': []}
        if s_mem:
            i_memory['a'].append(a)
            i_memory['b'].append(b)
            i_memory['q'].append(2)
            i_memory['Lq'].append(np.nan)
            i_memory['LTq'].append(np.nan)
        while (q >= qF) and (not stop):
            q = q - dq
            r = CalculateResidual(cx, cy, a, b)
            da, db = UpdateEstimate(cx, r, q, dq)
            a = a + da
            b = b + db
            y_hat = a * cx + b
            stop, c_lq, c_ltq = CheckStop(cy, y_hat, q, c_sigma, 1.1, True)
            if s_mem:
                i_memory['a'].append(a)
                i_memory['b'].append(b)
                i_memory['q'].append(q - dq)
                i_memory['Lq'].append(c_lq)
                i_memory['LTq'].append(c_ltq)
            q_final = q

        y_hat = a * cx + b
        Statistics = {'Residual': cy - y_hat, 'R2': R2(cy, y_hat), 'RMSE': RMSE(cy, y_hat),
                      'q_final': q_final, 'Lq_final': Lq(cy, y_hat, q), 'Iterations': iterations}

        return a, b, y_hat, Statistics, i_memory

    @staticmethod
    def PlotMemory(h, cx: np.ndarray, yGT: np.ndarray, yN: np.ndarray, s_memory: dict):
        c_N = len(s_memory['a'])

        for i in range(c_N):
            c_a = s_memory['a'][i]
            c_b = s_memory['b'][i]
            plt.plot(np.squeeze(cx), c_a * np.squeeze(cx) + c_b, ls='--', color=[0, i/c_N, (c_N - i)/c_N])

        h.plot(np.squeeze(cx), np.squeeze(yGT), '-k', label='GT')
        h.plot(cx, yN, '*r', label='Noisy')
        h.grid(True)
        plt.xlabel('$A_1$')
        plt.ylabel('b')


if __name__ == "__main__":
    i_q_final = 1.05
    m_sigma = 30
    Line.OUTLIERS = OutlierType.CONSTANT
    cl = Line(100, [0, 200], m_sigma, 5)
    x0, y0, yn = cl.Generate(3, -20)
    aH, bH, yH, iStatistics, mem_o = LqFitting.SolveLq(x0, yn, i_q_final, s_mem=True)
    aO, bO = OLLS(x0, yn)

    m_q_final = iStatistics['q_final']
    plt.figure(figsize=(8, 5))
    ch = plt.subplot(1, 2, 1)
    LqFitting.PlotMemory(ch, x0, y0, yn, mem_o)

    x0 = np.squeeze(x0)
    y0 = np.squeeze(y0)
    yn = np.squeeze(yn)
    yH = np.squeeze(yH)
    yO = np.squeeze(aO * x0 + bO)

    plt.subplot(1, 2, 2)
    plt.plot(x0, y0, '-k', label='GT')
    plt.plot(x0, yn, '*r', label='Noisy')
    plt.plot(x0, yH, '--g', label='Fit')
    plt.plot(x0, yO, '--b', label='OLLS')
    plt.grid(True)
    plt.xlabel('$A_1$')
    plt.legend()
    plt.title('$q_F$ = ' + str(np.round(m_q_final, 3)))
    plt.show()
