import numpy as np

class JD:
    def __init__(self, S0, data, mu, sigma, dt):
        self.S0 = S0
        self.mu = mu 
        self.sigma = sigma
        self.dt = dt

        self.drift = mu - 0.5 * sigma**2

        jump_mask = np.abs(data - self.mu) > 3 * self.sigma * np.sqrt(self.dt)

        num_jumps = int(jump_mask.sum())

        n = len(data)

        self.lam = num_jumps / (n * self.dt)  if num_jumps > 0 else 0

        if num_jumps > 1:
            Y = data[jump_mask] - self.drift * self.dt
            self.mu_j = Y.mean()
            self.sigma_j = Y.std()
        else:
            self.mu_j = 0
            self.sigma_j = 0

        data_nojump = data[~jump_mask]
        if len(data_nojump > 10):
            self.sigma = data_nojump.std() / np.sqrt(self.dt)
            self.drift = data_nojump.mean() - 0.5**self.sigma 

            self.mu = self.drift + 0.5 * self.sigma**2 # do not need 

    def sim(self, T):
        path = []
        S = self.S0
        for _ in range(T):
            Z = np.random.randn()
            N = np.random.poisson(self.lam * self.dt)

            total_jump = 0 
            for _ in range(N):
                total_jump += np.random.normal(self.mu_j, self.sigma_j)


            S *= np.exp(self.drift * self.dt + self.sigma * np.sqrt(self.dt) * Z + total_jump)
            path.append(S)

        return path