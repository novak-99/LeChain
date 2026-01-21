import numpy as np

class OU:
    def __init__(self, S0, data, dt):
        y = data[:-1]
        X = data[1:]

        self.b_hat = np.cov(X, y)[0, 1] / np.var(X)
        self.a_hat = np.mean(y) - self.b_hat * np.mean(X)

        y_hat = self.a_hat + X * self.b_hat

        var_eps = np.var(y - y_hat)

        self.kappa_hat = -np.log(self.b_hat) / dt

        self.theta_hat = self.a_hat / (1 - self.b_hat)

        self.sigma_hat = np.sqrt(var_eps * 2 * self.kappa_hat 
                                 / (1 - np.exp(-2 * self.kappa_hat * dt)))

        self.S0 = data[-1]

    def sim(self, T, dt):
        path = []
        S = self.S0
        for _ in range(T):
            Z = np.random.randn()
            S = self.theta_hat + (S - self.theta_hat) * np.exp(-self.kappa_hat * dt) + np.sqrt(self.sigma_hat**2 / (2 * self.kappa_hat) 
                      * (1 - np.exp(-2 * self.kappa_hat * dt))) * Z
            
            path.append(np.exp(S))

        return path