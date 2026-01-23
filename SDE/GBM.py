import numpy as np

class GBM:
    def __init__(self, S0, mu, sigma, dt):
        self.S0 = S0
        self.mu = mu 
        self.sigma = sigma
        self.dt = dt

        self.drift = mu - 0.5 * sigma**2

    def sim(self, T):
        path = []
        S = self.S0
        for _ in range(T):
            Z = np.random.randn()
            S *= np.exp(self.drift * self.dt + self.sigma * np.sqrt(self.dt) * Z)
            path.append(S)

        return path