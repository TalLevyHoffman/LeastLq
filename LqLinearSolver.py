from numba import njit
import numpy as np


@njit(parallel=False)
def RMSE(A: np.ndarray, x_hat: np.ndarray, b: np.ndarray) -> float: # noqa
    return np.sqrt(np.mean((b - A @ x_hat) ** 2))


@njit(parallel=False)
def Lq(cA: np.ndarray, x_hat: np.ndarray, cb: np.ndarray, q: float) -> float:
    return np.mean(np.abs(cb - cA @ x_hat) ** q)


class LqSolver:

    def __init__(self, A: np.ndarray, b: np.ndarray):
        self.A = A
        self.b = b
        self.IATA = np.linalg.inv(A.transpose() @ A)

    def L2Solver(self):
        ATA = self.A.transpose() @ self.A
        ATb = self.A.transpose() @ self.b
        x = np.linalg.solve(ATA, ATb)
        return x

    def __CalcResidual(self, x: np.ndarray):
        return self.A @ x - self.b

    def __UpdateEstimate(self, r, q, dq):
        eps = np.ones_like(r) * 1e-25
        dx = dq / (q - 1) * self.IATA @ self.A.transpose() @ (r * np.log(np.abs(r + eps)))
        return dx

    def Solve(self, q_final: float, iterations: int = 50):
        dq = (2.0 - q_final) / float(iterations)
        q = 2 + dq
        xL2 = self.L2Solver()
        x = np.copy(xL2)
        for i in range(iterations):
            q = q - dq
            r = self.__CalcResidual(x)
            dx = self.__UpdateEstimate(r, q, dq)
            x = x + dx

        r_final = self.__CalcResidual(x)
        Statistics = {'Residual': r_final, 'RMSE': RMSE(self.A, x, self.b),
                      'q_final': q_final, 'Lq_final': Lq(self.A, x, self.b, q_final), 'Iterations': iterations}

        return x, xL2, Statistics


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from IRLLS import IRLLS, WeightingType
    c_q_final = 1.1
    C_Lq11 = []
    C_Lq14 = []
    C_Lq17 = []
    C_IRLLS = []

    k_example = []

    for k in range(300):
        n = 160
        p = 5
        activate_noise = 40
        use_outliers = True
        c_A = np.random.randn(n, p) * 0.1
        x_gt = np.random.randn(p, 1)
        b_gt = c_A @ x_gt
        outlier_p = 0.3
        noise = activate_noise * np.linalg.norm(b_gt) / n / 10
        c_b = b_gt + np.random.randn(n, 1) * noise
        if use_outliers:
            no = int(np.round(b_gt.size * outlier_p, 0))
            Indices = np.random.permutation(n)
            c_b[Indices[0:no]] = np.random.rand(no, 1)*2

        lq = LqSolver(c_A, c_b)
        xH11, _, _ = lq.Solve(c_q_final, 80)

        solver = IRLLS(c_A, c_b, WeightingType.HUBER)
        solver.SetDefaultParams()
        xIRLLS = solver.Iterate()
        bIRLLS = c_A @ xIRLLS

        lq14 = LqSolver(c_A, c_b)
        xH14, _, _ = lq.Solve(1.4, 80)

        lq17 = LqSolver(c_A, c_b)
        xH17, _, _ = lq.Solve(1.7, 80)

        RMSE_Lq11 = np.sqrt(np.mean((x_gt - xH11) ** 2))
        RMSE_Lq14 = np.sqrt(np.mean((x_gt - xH14) ** 2))
        RMSE_Lq17 = np.sqrt(np.mean((x_gt - xH17) ** 2))
        RMSE_IRLLS = np.sqrt(np.mean((x_gt - xIRLLS) ** 2))

        C_Lq11.append(RMSE_Lq11)
        C_Lq14.append(RMSE_Lq14)
        C_Lq17.append(RMSE_Lq17)
        C_IRLLS.append(RMSE_IRLLS)

        if k in k_example:
            plt.figure(figsize=(14, 7))
            plt.subplot(1, 2, 1)
            plt.plot(x_gt, xH14, '*g', label='Lq_1.4')
            plt.plot(x_gt, xIRLLS, '*c', label='IRLLS-HUBER')
            plt.plot(x_gt, x_gt, '*k', label='GT')
            plt.grid(True)
            plt.xlabel("$X_{GT}$")
            plt.ylabel('X')
            plt.legend()
            plt.subplot(1, 2, 2)
            plt.plot(b_gt, c_b, '*b', label='Noisy')
            plt.plot(b_gt, c_A @ xH14, '*g', label='Lq_1.4')
            plt.plot(b_gt, bIRLLS, '*c', label='IRLLS-HUBER')
            plt.plot(b_gt, b_gt, '*k', label='GT')
            plt.grid(True)
            plt.xlabel("$b_{GT}$")
            plt.ylabel('b')
            plt.legend()
            plt.suptitle('pO = ' + str(outlier_p * 100) + '[%], ' + r'$\sigma$ = ' + str(np.round(noise, 3))
                         + ", $qF_1.4$ = " + str(np.round(c_q_final, 3)) + r', $RMSE_{L_q_1.4}$ = '
                         + str(np.round(RMSE_Lq14, 3))
                         + r', $RMSE_{IRLLS}$ = ' + str(np.round(RMSE_IRLLS, 3)))

            plt.show()

    data = [np.asfarray(C_Lq11), np.asfarray(C_Lq14), np.asfarray(C_Lq17), np.asfarray(C_IRLLS)]
    plt.figure()
    plt.boxplot(data, labels=['Lq-' + str(c_q_final), 'Lq-1.4', 'Lq-1.7', 'IRLLS-Huber'])
    plt.ylabel('RMSE')
    plt.title(r'$\sigma$=0.1, pO=0%')
    plt.grid(True)
    plt.show()
