from abc import ABC
from datetime import datetime, timezone

import time, requests
import pandas as pd

import numpy as np
from SDE.GBM import GBM
from SDE.OU import OU

from sklearn.linear_model import Ridge

class Coin(ABC):
    def __init__(self, method="gbm", freq="hourly"):
        self.freq = freq
        self.method = method
        self.dt = self.freq_to_dt(self.freq)
        self.download_hist()

        if method == "gbm": self.handle_gbm()
        elif method == "ou": self.handle_ou()
        elif method == "ridge": self.handle_ridge()

    def handle_gbm(self):
        r = np.diff(np.log(self.data))

        self.r_hat = r.mean()
        self.s = r.std()

        self.sigma = np.sqrt(self.s**2 / self.dt)
        self.mu = self.r_hat / self.dt + 0.5 * self.sigma**2

        self.gbm = GBM(self.data[-1], self.mu, self.sigma)

    def handle_ou(self):
        r = np.log(self.data)
        self.ou = OU(self.data[-1], r, self.dt)

    def handle_ridge(self):
        self.ridge = Ridge(alpha=1.0)
        self.W = 24

        # rmr to make into Ridge object.
        self.r = np.diff(np.log(self.data))
        X, y = self.make_dataset(self.r, self.W)
        self.ridge.fit(X, y)

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

    def sim(self, T, freq="hourly"):
        # NOTE : ADD HANDLE SIM FUNCTIONS, OR ADD RIDGE + ABSTRACT ML PIPELINE.
        if self.method == "gbm": return self.gbm.sim(T, self.freq_to_dt(freq))
        elif self.method == "ou": return self.ou.sim(T, self.freq_to_dt(freq)) 
        elif self.method == "ridge": 
            # seed -> last W returns
            state = self.r[-self.W:].copy() # dc

            path = []

            S = self.data[-1]

            for _ in range(T):
                y_hat = float(self.ridge.predict(state.reshape(1, -1))[0])

                S *= np.exp(y_hat)
                path.append(S)

                state = np.roll(state, -1) # move s0 to front to replace
                state[-1] = y_hat
                 
            return path

    
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
    
    def make_dataset(self, r, W):
        X, y = [], []

        for i in range(W, len(r)):
            X.append(r[i - W:i])
            y.append(r[i])

        return (np.array(X), np.array(y))