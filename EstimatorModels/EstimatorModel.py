from abc import ABC, abstractmethod
import numpy as np
from sklearn.linear_model import Lasso

class EstimatorModel(ABC):
    @abstractmethod
    def sim(self, T):
        pass

    def make_dataset(self, r, W):
        X, y = [], []

        for i in range(W, len(r)):
            X.append(r[i - W:i])
            y.append(r[i])

        return (np.array(X), np.array(y))