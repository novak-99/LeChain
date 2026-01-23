# LeChain

SDE/ML solver package for cypto price estimation. Please see main.py for an example using ETH.

## Setup 

Create a Python environment for the project and activate:

```bash
python3 -m venv venv
source venv/bin/activate
```

Then, install the necessary dependencies: 

```bash
pip install -r requirements.txt
```

## Usage

Choose a coin to simulate and specify a frequency and method (default is Geometric Brownian Motion):

```py
from Coins.BTC import BTC # simulate BTC

coin = BTC(method="gbm", freq="hourly")

S = coin.sim(T=1000) # generate a path
```

## Features 

### Coins
1. BTC
2. LTC
3. ETH

### Models

#### SDE Solvers

1. Geometric Brownian Motion
2. Ornstein–Uhlenbeck
3. Jump Diffusion
4. Heston

#### Machine Learning Models

1. Lasso Regression
2. Ridge Regression
3. Trees (XGBoost)
4. Artifical Neural Network
5. LSTM 
6. Transformer