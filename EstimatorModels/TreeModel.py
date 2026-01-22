import numpy as np
import xgboost as xgb
from EstimatorModels.EstimatorModel import EstimatorModel

class TreeModel(EstimatorModel):
    def __init__(self, data, W = 24):
        self.model = xgb.XGBRegressor(n_estimators=1000, 
                                 max_depth=6, learning_rate=0.05)
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