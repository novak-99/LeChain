from Coins.LTC import LTC

coin = LTC(method="ou")

S = coin.sim(T=1000, freq="hourly")

print(S)