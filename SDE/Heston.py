import numpy as np

class Heston:
    def __init__(self, S0, r, dt, lam=0.94, tol=1e-12):
        self.S0 = S0
        self.dt = dt
        self.tol = tol

        inst = (r * r) / self.dt

        # too noisy without this. you can't properly run with current setup.
        # smoothing.
        v = np.empty_like(inst)
        v[0] = max(inst[0], self.tol)
        for t in range(1, inst.size):
            v[t] = lam * v[t - 1] + (1.0 - lam) * max(inst[t], 0.0)
        v = np.maximum(v, self.tol)

        self.v0 = v[-1]

        X = v[:-1]
        y = v[1:]

        b_hat = np.cov(X, y)[0, 1] / np.var(X)

        a_hat = np.mean(y) - b_hat * np.mean(X)

        self.kappa = -np.log(b_hat) / self.dt
        self.theta = max(a_hat / (1.0 - b_hat), self.tol)

        eps = y - (a_hat + b_hat * X)

        Xt = np.maximum(X, self.tol)
        xi2 = np.mean((eps * eps) / (Xt * self.dt))
        self.xi = np.sqrt(max(xi2, self.tol))

        self.mu = np.mean(r) / self.dt + 0.5 * np.mean(v)

        vt = np.maximum(v, self.tol)
        Z1 = ((r - (self.mu - 0.5 * vt) * self.dt) / np.sqrt(vt * self.dt))[1:]
        Z2 = eps / (self.xi * np.sqrt(Xt * self.dt))

        rho = np.corrcoef(Z1, Z2)[0, 1]
        self.rho = np.clip(rho, -0.999, 0.999)

    def sim(self, T):

        path = []
        S = self.S0
        v = self.v0

        for i in range(int(T)):
            Z1 = np.random.randn()
            Zp = np.random.randn()
            Z2 = self.rho * Z1 + np.sqrt(max(1.0 - self.rho * self.rho, 1e-12)) * Zp

            v_pos = max(v, 0.0)
            v = v + self.kappa * (self.theta - v_pos) * self.dt + self.xi * np.sqrt(v_pos) * np.sqrt(self.dt) * Z2
            v = max(v, 0.0)

            # Price update
            S *= np.exp((self.mu - 0.5 * v) * self.dt + np.sqrt(max(v, 0.0) * self.dt) * Z1)
            path.append(S)

        return path
