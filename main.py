from Coins.LTC import LTC

coin = LTC(method="ann")

# S = coin.sim(T=1000, freq="hourly")

S = coin.sim(T=1000)

print(S)