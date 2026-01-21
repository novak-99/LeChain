from EstimatorModels.EstimatorModel import EstimatorModel
import torch
import torch.nn as nn
import numpy as np

class TorchANN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(TorchANN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU() # Activation function
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

class ANN(EstimatorModel):
    def __init__(self, data, W = 24):

        # lgtm
        epochs = 1_000

        self.model = TorchANN(W, W//2, 1).double()
        self.W = W

        self.r = torch.from_numpy(np.diff(np.log(data)))
        X, y = self.make_dataset(self.r, self.W)

        X, y = torch.from_numpy(X), torch.from_numpy(y)

        criterion = nn.MSELoss()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)

        # Mini-training loop
        for _ in range(epochs):
            # Forward pass
            outputs = self.model(X)
            loss = criterion(outputs, y)

            # Backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        self.S0 = data[-1]

    def sim(self, T):
        # seed -> last W returns
        state = self.r[-self.W:].clone() # dc

        path = []

        S = self.S0

        for _ in range(T):
            y_hat = float(self.model(state.reshape(1, -1))[0])

            S *= np.exp(y_hat)
            path.append(S)

            state = torch.roll(state, -1) # move s0 to front to replace
            state[-1] = y_hat
            
        return path