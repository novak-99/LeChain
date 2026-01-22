import torch 
import torch.nn as nn

import numpy as np

from EstimatorModels.EstimatorModel import EstimatorModel

class TorchTransformer(nn.Module):

    def __init__(self, d_model=64, nhead=4, num_layers=2, 
                 dim_feedforward=128, dropout=0.1, max_len=4096):
        
        super().__init__()
        self.in_proj = nn.Linear(1, d_model)

        self.pos_emb = nn.Embedding(max_len, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True, 
            norm_first=True,
            dropout=dropout
        )

        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.out = nn.Linear(d_model, 1)

    def forward(self, x):
        B, W, _ = x.shape
        h = self.in_proj(x)

        pos = torch.arange(W).unsqueeze(0).expand(B, W)
        h = h + self.pos_emb(pos)

        causal_mask = torch.triu(torch.ones(W, W, dtype=torch.bool), diagonal=1)

        z = self.encoder(h, mask=causal_mask)
        y = self.out(z[:, -1, :])
        return y
    
class Transformer(EstimatorModel):
    def __init__(self, data, W = 24):

        # long, but ensures proper training.
        epochs = 1000

        self.model = TorchTransformer().double()
        self.W = W

        self.r = torch.from_numpy(np.diff(np.log(data)))
        X, y = self.make_dataset(self.r, self.W)


        X, y = torch.from_numpy(X).reshape(-1, self.W, 1), torch.from_numpy(y).reshape(-1, 1)


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
            y_hat = float(self.model(state.view(1, self.W, 1))[0])

            S *= np.exp(y_hat)
            path.append(S)

            state = torch.roll(state, shifts=-1, dims=0) # move s0 to front to replace
            state[-1] = y_hat
            
        return path
