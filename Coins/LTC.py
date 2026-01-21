from Coins.Coin import Coin

class LTC(Coin):
    url = "https://api.exchange.coinbase.com/products/LTC-USD/candles"
    def __init__(self, method="gbm", freq="hourly"):
        super().__init__(method=method, freq=freq)