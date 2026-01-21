from Coins.LTC import LTC

coin = LTC()

S = coin.sim(T=1000, freq="hourly")

print(S)