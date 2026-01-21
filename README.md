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
