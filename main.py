from Coins.LTC import LTC

coin = LTC(method="ridge")

# S = coin.sim(T=1000, freq="hourly")

S = coin.sim(T=1000)

print(S)