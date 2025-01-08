import numpy as np


class SuperEllipsoid:

    def __init__(self, a, b, c, e1, e2):
        self.a = a
        self.b = b
        self.c = c
        self.e1 = e1
        self.e2 = e2
        self.points = None
        self.m = "gist_rainbow"

    def sample_fibonacci(self, num_samples=2500, max_iter=6, limit_e1_e2=True):
        if limit_e1_e2:
            self.e1 = np.clip(self.e1, 0.01, 10.0)
            self.e2 = np.clip(self.e2, 0.01, 10.0)

        phi = np.pi * (3 - np.sqrt(5))        # golden angle in radians
        i = np.arange(num_samples)
        y = 1 - (i / (num_samples - 1)) * 2   # y goes from 1 to -1
        radius = np.sqrt(1 - y * y)           # radius at y
        theta = phi * i                       # golden angle increment
        x = np.cos(theta) * radius
        z = np.sin(theta) * radius
        s = np.array([x, y, z]).T

        ToE1 = 2. / self.e1
        ToE2 = 2. / self.e2
        Rat = self.e2/self.e1

        for i in range(max_iter):
            f = (np.abs(s[:, 0])**ToE2 + np.abs(s[:, 1])**ToE2)**Rat + np.abs(s[:, 2])**ToE1
            d = np.sign(f-1) * np.abs(1 - f**(self.e1/2))
            m = np.linalg.norm(s, axis=1)
            s *= (1-(d/m)*(m-d)).reshape(-1, 1)

        self.points = s * np.array([self.a, self.b, self.c])

    def Show(self):
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_title("$a={}$, $b={}$, $c={}$, $e_1={}$, $e_2={}$".format(self.a, self.b, self.c, self.e1, self.e2))
        _ = ax.scatter(self.points[:, 0], self.points[:, 1], self.points[:, 2], c=-self.points[:, 2], cmap=self.m)
        plt.show()


if __name__ == "__main__":
    R = 1
    c_e = 2
    s_ell = SuperEllipsoid(R, R, R, c_e, c_e)
    s_ell.sample_fibonacci()
    s_ell.Show()
