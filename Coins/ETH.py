from Coins.Coin import Coin

class ETH(Coin):
    url = "https://api.exchange.coinbase.com/products/ETH-USD/candles"
    def __init__(self, method="gbm", freq="hourly"):
        super().__init__(method=method, freq=freq)