from abc import ABC, abstractmethod
from datetime import datetime, timezone

import time, requests
import pandas as pd

import numpy as np
from SDE.GBM import GBM

class Coin(ABC):
    def download_hist(self):
        LOOKBACK = 86400 * 90  # := 90 days
        CB_MAX = 300

        now = int(time.time())
        start_ts = now - LOOKBACK
        end_ts = now

        gran = self.freq_to_gran(self.freq)

        # BTC placeholder.
        # url = "https://api.exchange.coinbase.com/products/BTC-USD/candles"
        headers = {"User-Agent": "LeChain/0.1"}

        step = gran * CB_MAX

        data = []
        t = int(start_ts)
        while t < end_ts:
            t2 = min(t + step, end_ts)
            params = {
                "start": self.to_iso_z(t),
                "end": self.to_iso_z(t2),
                "granularity": gran,
            }

            resp = requests.get(self.url, params=params, headers=headers, timeout=20)
            if resp.status_code != 200:
                raise RuntimeError(f"Error while getting data for {self.__class__.__name__} {resp.status_code}: {resp.text[:300]}")

            data.extend(resp.json())
            t = t2

        # rows := [time, low, high, open, close, volume]
        df = pd.DataFrame(data, columns=["ts", "low", "high", "open", "close", "volume"])

        df = df.drop_duplicates(subset=["ts"]).sort_values("ts").reset_index(drop=True)

        # np array
        self.data = df["close"].astype(float).to_numpy()

    def __init__(self, method="gbm", freq="hourly"):
        self.freq = freq
        self.download_hist()

        r = np.diff(np.log(self.data))

        self.r_hat = r.mean()
        self.s = r.std()

        self.dt = self.freq_to_dt(freq)
        self.sigma = np.sqrt(self.s**2 / self.dt)
        self.mu = self.r_hat / self.dt + 0.5 * self.sigma**2

        self.gbm = GBM(self.data[-1], self.mu, self.sigma)


    def sim(self, T, freq):
        return self.gbm.sim(T, self.freq_to_dt(freq))
    
    def freq_to_dt(self, freq):
        dt = 1/365
        if freq == "daily": pass
        elif freq == "hourly": dt /= 24
        elif freq == "minute": dt /= (24 * 60)
        elif freq == "second": dt /= (24 * 60 * 60)
        else: raise ValueError(f"Unknown frequency type: \'{freq}\'")
        return dt
    
    def freq_to_gran(self, freq):
        gran = 1
        if freq == "daily": gran *= 86400
        elif freq == "hourly": gran *= 3600
        elif freq == "minute": gran *= 60
        elif freq == "second": pass
        else: raise ValueError(f"Unknown freq type: \'{freq}\'")
        return gran
    
    def to_iso_z(self, ts):
        return datetime.fromtimestamp(int(ts), tz=timezone.utc).isoformat().replace("+00:00", "Z")