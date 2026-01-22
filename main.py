from Coins.LTC import LTC
from Coins.ETH import ETH

coin = ETH(method="tree")

# S = coin.sim(T=1000, freq="hourly")

S = coin.sim(T=1000)

print(S)