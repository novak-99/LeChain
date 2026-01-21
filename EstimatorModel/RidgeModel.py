import numpy as np
from sklearn.linear_model import Ridge

class RidgeModel:
    def __init__(self, data, W = 24):
        self.model = Ridge(alpha=1.0)
        self.W = W

        self.r = np.diff(np.log(data))
        X, y = self.make_dataset(self.r, self.W)
        self.model.fit(X, y)

        self.S0 = data[-1]

    def sim(self, T):
        # seed -> last W returns
        state = self.r[-self.W:].copy() # dc

        path = []

        S = self.S0

        for _ in range(T):
            y_hat = float(self.model.predict(state.reshape(1, -1))[0])

            S *= np.exp(y_hat)
            path.append(S)

            state = np.roll(state, -1) # move s0 to front to replace
            state[-1] = y_hat
            
        return path

    def make_dataset(self, r, W):
        X, y = [], []

        for i in range(W, len(r)):
            X.append(r[i - W:i])
            y.append(r[i])

        return (np.array(X), np.array(y))