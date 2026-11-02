"""Vectorized technical indicator calculations used by the PowerLanguage runtime.

This module implements common bar-based studies. It is deliberately independent
of order simulation and only uses information available through the requested
bar index. MultiCharts' licensed studies and chart engine are not reimplemented.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd


class _PLMedian:
    field = "__median_price__"


class IndicatorLibrary:
    """Lazy, cached OHLCV indicator series with PowerLanguage-style inputs."""

    def __init__(self, bars, session_timezone="UTC"):
        self.bars = bars
        self.session_timezone = session_timezone
        self.cache: dict[tuple, np.ndarray] = {}
        self._source_refs: dict[int, np.ndarray] = {}
        self.o = bars["open"].to_numpy(dtype=np.float64)
        self.h = bars["high"].to_numpy(dtype=np.float64)
        self.l = bars["low"].to_numpy(dtype=np.float64)
        self.c = bars["close"].to_numpy(dtype=np.float64)
        self.v = bars["volume"].to_numpy(dtype=np.float64)
        self.fields = {str(field).lower(): pd.to_numeric(bars[field], errors="coerce").to_numpy(dtype=np.float64)
                       for field in bars.columns if field != "timestamp"}
        self.fields.update({"open": self.o, "high": self.h, "low": self.l, "close": self.c, "volume": self.v})
        self.fields["__median_price__"]=(self.h+self.l)/2
        try:
            stamps = pd.DatetimeIndex(bars.timestamp)
            if stamps.tz is not None:
                stamps = stamps.tz_convert(session_timezone)
            self.session_dates = np.asarray(stamps.strftime("%Y%m%d"), dtype=np.int64)
            self.week_keys = np.asarray(stamps.strftime("%G%V"), dtype=np.int64)
            self.month_keys = np.asarray(stamps.strftime("%Y%m"), dtype=np.int64)
            self.year_keys = np.asarray(stamps.year, dtype=np.int64)
        except Exception:
            self.session_dates = np.arange(len(bars), dtype=np.int64)
            self.week_keys = self.month_keys = self.year_keys = self.session_dates.copy()
        self._group_cache = {}

    @staticmethod
    def _field(value, default="close"):
        return getattr(value, "field", default)

    def _src(self, value=None, default="close"):
        field = self._field(value, default)
        fallback=np.full(len(self.c),np.nan,dtype=np.float64) if field.endswith("_data2") else self.c
        return np.asarray(self.fields.get(field, fallback), dtype=np.float64), field

    def _grouped(self, field, group_keys, operation="last", cumulative=False):
        key = ("grouped", field, id(group_keys), operation, cumulative)
        if key not in self.cache:
            values = self.fields.get(field, self.c)
            frame = pd.DataFrame({"group": group_keys, "value": values})
            groups = frame.groupby("group", sort=False)["value"]
            if cumulative:
                result = groups.cummax() if operation == "max" else groups.cummin() if operation == "min" else groups.cumsum()
            elif operation == "first":
                result = groups.transform("first")
            elif operation == "max":
                result = groups.transform("max")
            elif operation == "min":
                result = groups.transform("min")
            elif operation == "sum":
                result = groups.transform("sum")
            else:
                result = groups.transform("last")
            self.cache[key] = result.to_numpy(dtype=np.float64)
        return self.cache[key]

    def _prior_daily(self, field):
        key = ("prior_daily", field)
        if key not in self.cache:
            frame = pd.DataFrame({"date": self.session_dates, "value": self.fields.get(field, self.c)})
            daily = frame.groupby("date", sort=False).agg(
                value=("value", "first" if field == "open" else "max" if field == "high" else
                       "min" if field == "low" else "sum" if field == "volume" else "last")
            )
            previous = daily.value.shift(1)
            mapping = previous.to_dict()
            self.cache[key] = frame.date.map(mapping).to_numpy(dtype=np.float64)
        return self.cache[key]

    def _rolling(self, values, length, op, min_periods=None):
        length = max(1, int(length))
        self._source_refs[id(values)] = values
        key = ("rolling", id(values), length, op, min_periods)
        if key not in self.cache:
            s = pd.Series(values)
            win = s.rolling(length, min_periods=min_periods or length)
            funcs = {"mean": win.mean, "sum": win.sum, "max": win.max, "min": win.min,
                     "std": lambda: win.std(ddof=0), "std_sample": lambda: win.std(ddof=1)}
            self.cache[key] = funcs[op]().to_numpy(dtype=np.float64)
        return self.cache[key]

    def _ema(self, values, length, seed="first"):
        length = max(1, int(length))
        self._source_refs[id(values)] = values
        key = ("ema", id(values), length, seed)
        if key not in self.cache:
            arr = np.asarray(values, dtype=np.float64)
            out = np.full(len(arr), np.nan)
            valid = np.flatnonzero(np.isfinite(arr))
            if len(valid):
                start = int(valid[0])
                alpha = 2.0 / (length + 1.0)
                if seed == "sma" and start + length <= len(arr):
                    out[start + length - 1] = np.mean(arr[start:start + length])
                    first = start + length
                else:
                    out[start] = arr[start]
                    first = start + 1
                for j in range(first, len(arr)):
                    if np.isfinite(arr[j]):
                        prev = out[j - 1]
                        out[j] = arr[j] if not np.isfinite(prev) else alpha * arr[j] + (1-alpha) * prev
                if seed == "sma" and first > start + length:
                    out[:start + length - 1] = np.nan
            self.cache[key] = out
        return self.cache[key]

    def _wilder(self, values, length):
        length = max(1, int(length))
        self._source_refs[id(values)] = values
        key = ("wilder", id(values), length)
        if key not in self.cache:
            arr = np.asarray(values, dtype=np.float64)
            out = np.full(len(arr), np.nan)
            if len(arr) >= length:
                out[length - 1] = np.nanmean(arr[:length])
                for j in range(length, len(arr)):
                    out[j] = (out[j-1] * (length - 1) + arr[j]) / length
            self.cache[key] = out
        return self.cache[key]

    def _tr(self):
        key = ("true_range",)
        if key not in self.cache:
            pc = np.r_[np.nan, self.c[:-1]]
            self.cache[key] = np.nanmax(np.vstack((self.h-self.l, np.abs(self.h-pc), np.abs(self.l-pc))), axis=0)
        return self.cache[key]

    def _rsi(self, values, length):
        length = max(1, int(length))
        key = ("rsi", id(values), length)
        if key not in self.cache:
            result = np.full(len(values), np.nan)
            change = np.diff(values)
            gain, loss = np.maximum(change, 0), np.maximum(-change, 0)
            if len(values) > length:
                avg_gain, avg_loss = float(np.mean(gain[:length])), float(np.mean(loss[:length]))
                for j in range(length, len(values)):
                    if j > length:
                        avg_gain = (avg_gain * (length-1) + gain[j-1]) / length
                        avg_loss = (avg_loss * (length-1) + loss[j-1]) / length
                    if avg_gain == 0 and avg_loss == 0:
                        result[j] = 50
                    elif avg_loss == 0:
                        result[j] = 100
                    elif avg_gain == 0:
                        result[j] = 0
                    else:
                        result[j] = 100 - 100 / (1 + avg_gain / avg_loss)
            self.cache[key] = result
        return self.cache[key]

    def _di_adx(self, length):
        length = max(1, int(length))
        key = ("di_adx", length)
        if key not in self.cache:
            up = self.h - np.r_[np.nan, self.h[:-1]]
            down = np.r_[np.nan, self.l[:-1]] - self.l
            plus = np.where((up > down) & (up > 0), up, 0.0)
            minus = np.where((down > up) & (down > 0), down, 0.0)
            atr = self._wilder(self._tr(), length)
            plus_di = np.divide(100 * self._wilder(plus, length), atr,
                                out=np.full(len(atr), np.nan), where=atr != 0)
            minus_di = np.divide(100 * self._wilder(minus, length), atr,
                                 out=np.full(len(atr), np.nan), where=atr != 0)
            den = plus_di + minus_di
            dx = np.divide(100 * np.abs(plus_di-minus_di), den,
                           out=np.full(len(den), np.nan), where=den != 0)
            self.cache[key] = (plus_di, minus_di, self._wilder(dx, length))
        return self.cache[key]

    def series(self, name, *args):
        name = name.lower()
        # Price argument, then integer periods and optional numeric settings.
        src_value = args[0] if args else None
        src, field = self._src(src_value)
        nums = [float(x) for x in args
                if isinstance(x, (int, float, np.number)) and not hasattr(x, "field")]
        ints = [max(1, int(x)) for x in nums]
        length = ints[0] if ints else 14
        cache_key = (name, field, tuple(nums))
        if cache_key in self.cache:
            return self.cache[cache_key]
        c, h, l, o, v = self.c, self.h, self.l, self.o, self.v
        n = len(c)
        tr = self._tr()
        out = np.full(n, np.nan)

        if name in {"sma", "simplemovingaverage", "movavgsimple", "average", "averagefc"}:
            out = self._rolling(src, length, "mean")
        elif name in {"ema", "exponentialmovingaverage", "movavgexponential", "xaverage"}:
            out = self._ema(src, length)
        elif name in {"wma", "weightedaverage", "weightedmovingaverage", "movavgweighted"}:
            weights = np.arange(1, length+1, dtype=float)
            out = np.convolve(np.nan_to_num(src), weights[::-1], mode="full")[:n] / weights.sum()
            out[:length-1] = np.nan
        elif name in {"trima", "triangularmovingaverage", "movavgtriangular"}:
            out = self._rolling(self._rolling(src, max(1, math.ceil(length/2)), "mean"),
                                max(1, math.floor(length/2)+1), "mean")
        elif name in {"hma", "hullmovingaverage", "movavghull"}:
            half, root = max(1, length//2), max(1, int(math.sqrt(length)))
            def wma(x, p):
                weights = np.arange(1, p+1, dtype=float)
                return np.convolve(np.nan_to_num(x), weights[::-1], mode="full")[:n] / weights.sum()
            diff = 2*wma(src, half)-wma(src, length)
            out = wma(diff, root); out[:length-1] = np.nan
        elif name in {"tr", "truerange"}:
            out = tr
        elif name in {"atr", "avgtruerange", "averagetruerange"}:
            out = self._wilder(tr, length)
        elif name in {"wildersaverage", "wilderaverage"}:
            out = self._wilder(src, length)
        elif name in {"stddev", "standarddev", "stdev", "volatilitystddev"}:
            out = self._rolling(src, length, "std")
        elif name in {"variance", "var"}:
            sd = self._rolling(src, length, "std")
            out = sd*sd
        elif name in {"momentum", "mom"}:
            out[length:] = src[length:] - src[:-length]
        elif name in {"roc", "rateofchange", "volumerateofchange"}:
            base = v if name == "volumerateofchange" else src
            out[length:] = np.divide(base[length:]-base[:-length], base[:-length],
                                     out=np.full(n-length, np.nan), where=base[:-length] != 0)*100
        elif name in {"rsi", "relativestrengthindex"}:
            out = self._rsi(src, length)
        elif name in {"cci", "commoditychannelindex", "commchannelindex"}:
            tp = (h+l+c)/3
            mean = self._rolling(tp, length, "mean")
            deviations = np.full(n, np.nan)
            if n >= length:
                windows = np.lib.stride_tricks.sliding_window_view(tp, length)
                chunk = max(1, min(len(windows), 100_000 // length))
                for start in range(0, len(windows), chunk):
                    stop = min(len(windows), start+chunk)
                    deviations[start+length-1:stop+length-1] = np.mean(
                        np.abs(windows[start:stop]-mean[start+length-1:stop+length-1, None]), axis=1
                    )
            out = np.divide(tp-mean, .015*deviations, out=out, where=deviations != 0)
        elif name in {"adx", "averagedirectionalindex"}:
            out = self._di_adx(length)[2]
        elif name in {"plusdi", "diplus", "positive_directional_indicator"}:
            out = self._di_adx(length)[0]
        elif name in {"minusdi", "diminus", "negative_directional_indicator"}:
            out = self._di_adx(length)[1]
        elif name == "adxr":
            adx = self._di_adx(length)[2]; lag = max(1, length)
            out[lag:] = (adx[lag:]+adx[:-lag])/2
        elif name in {"macd", "macdline"}:
            fast, slow = (ints[:2] if len(ints) >= 2 else (12, 26))
            out = self._ema(src, fast)-self._ema(src, slow)
        elif name in {"macdsignal", "macdtrigger"}:
            fast, slow, sig = (ints + [12, 26, 9])[:3]
            out = self._ema(self._ema(src, fast)-self._ema(src, slow), sig)
        elif name in {"macdhistogram", "macdhist"}:
            fast, slow, sig = (ints + [12, 26, 9])[:3]
            line = self._ema(src, fast)-self._ema(src, slow)
            out = line-self._ema(line, sig)
        elif name in {"bollingerupper", "bollingerbandupper", "bbupper", "bollingerband"}:
            mult = nums[1] if len(nums) > 1 else 2.0
            out = self._rolling(src, length, "mean") + mult*self._rolling(src, length, "std")
        elif name in {"bollingerlower", "bollingerbandlower", "bblower"}:
            mult = nums[1] if len(nums) > 1 else 2.0
            out = self._rolling(src, length, "mean") - mult*self._rolling(src, length, "std")
        elif name in {"bollingerpercentb", "percentb"}:
            mean, sd = self._rolling(src, length, "mean"), self._rolling(src, length, "std")
            out = np.divide(src-(mean-2*sd), 4*sd, out=out, where=sd != 0)
        elif name in {"bollingerbandwidth", "bandwidth"}:
            mean, sd = self._rolling(src, length, "mean"), self._rolling(src, length, "std")
            out = np.divide(4*sd, mean, out=out, where=mean != 0)*100
        elif name in {"fastk", "stochasticfastk", "slowk", "stochasticslowk", "stochastic"}:
            hh, ll = self._rolling(h, length, "max"), self._rolling(l, length, "min")
            raw = np.divide(100*(c-ll), hh-ll, out=out.copy(), where=hh != ll)
            smooth = ints[1] if len(ints) > 1 else 3
            out = self._rolling(raw, smooth, "mean") if name in {"slowk", "stochasticslowk"} else raw
        elif name in {"fastd", "stochasticfastd", "slowd", "stochasticslowd"}:
            hh, ll = self._rolling(h, length, "max"), self._rolling(l, length, "min")
            raw = np.divide(100*(c-ll), hh-ll, out=out.copy(), where=hh != ll)
            smooth_k = ints[1] if len(ints) > 1 else 3
            k = self._rolling(raw, smooth_k, "mean") if name in {"slowd", "stochasticslowd"} else raw
            smooth_d = ints[2] if len(ints) > 2 else 3
            out = self._rolling(k, smooth_d, "mean")
        elif name in {"williamsr", "williamspercentR", "percentR".lower()}:
            hh, ll = self._rolling(h, length, "max"), self._rolling(l, length, "min")
            out = np.divide(-100*(hh-c), hh-ll, out=out, where=hh != ll)
        elif name in {"obv", "onbalancevolume"}:
            direction = np.sign(np.diff(c, prepend=c[0])); out = np.cumsum(direction*v)
        elif name in {"accumdist", "accumulationdistribution", "chaikinad"}:
            mfm = np.divide((c-l)-(h-c), h-l, out=np.zeros(n), where=h != l)
            out = np.cumsum(mfm*v)
        elif name in {"chaikinosc", "chaikino scillator".replace(" ", "")}:
            mfm = np.divide((c-l)-(h-c), h-l, out=np.zeros(n), where=h != l)
            ad = np.cumsum(mfm*v); fast, slow = (ints[:2] if len(ints) >= 2 else (3, 10))
            out = self._ema(ad, fast)-self._ema(ad, slow)
        elif name in {"mfi", "moneyflowindex"}:
            tp = (h+l+c)/3; flow = tp*v; delta = np.diff(tp, prepend=tp[0])
            pos = self._rolling(np.where(delta > 0, flow, 0), length, "sum")
            neg = self._rolling(np.where(delta < 0, flow, 0), length, "sum")
            valid=np.isfinite(pos)&np.isfinite(neg)
            out=np.full(n,np.nan)
            out[valid & (pos==0) & (neg==0)]=50
            out[valid & (neg==0) & (pos>0)]=100
            active=valid & (neg>0)
            out[active]=100-100/(1+pos[active]/neg[active])
        elif name in {"volumeoscillator", "volosc"}:
            fast, slow = (ints[:2] if len(ints) >= 2 else (5, 20))
            a, b = self._ema(v, fast), self._ema(v, slow)
            out = np.divide(a-b, b, out=out, where=b != 0)*100
        elif name in {"pvt", "pricevolumetrend"}:
            roc = np.divide(np.diff(c, prepend=c[0]), np.r_[np.nan, c[:-1]], out=np.zeros(n), where=np.r_[np.nan, c[:-1]] != 0)
            out = np.cumsum(np.nan_to_num(roc)*v)
        elif name == "trix":
            e1 = self._ema(src, length); e2 = self._ema(e1, length); e3 = self._ema(e2, length)
            out[1:] = np.divide(e3[1:]-e3[:-1], e3[:-1], out=np.full(n-1, np.nan), where=e3[:-1] != 0)*100
        elif name == "ppo":
            fast, slow = (ints[:2] if len(ints) >= 2 else (12, 26)); a, b = self._ema(src, fast), self._ema(src, slow)
            out = np.divide(a-b, b, out=out, where=b != 0)*100
        elif name in {"priceoscillator", "priceosc"}:
            fast, slow = (ints[:2] if len(ints) >= 2 else (10, 20)); out = self._ema(src, fast)-self._ema(src, slow)
        elif name in {"massindex", "mass"}:
            spread = h-l; e1=self._ema(spread, 9); ratio=np.divide(e1, self._ema(e1, 9), out=np.zeros(n), where=self._ema(e1,9)!=0)
            out = self._rolling(ratio, length, "sum")
        elif name in {"ultosc", "ultimateoscillator"}:
            pc=np.r_[c[0], c[:-1]]; bp=c-np.minimum(l,pc); tr_u=h-np.maximum(l,pc)
            p1,p2,p3=(ints+[7,14,28])[:3]
            def ratio(p):
                a=self._rolling(bp,p,"sum"); b=self._rolling(tr_u,p,"sum")
                return np.divide(a,b,out=np.zeros(n),where=b!=0)
            out=100*(4*ratio(p1)+2*ratio(p2)+ratio(p3))/7
        elif name in {"ulcerindex", "ulcer"}:
            peak=self._rolling(src,length,"max"); dd=np.divide(src-peak,peak,out=np.zeros(n),where=peak!=0)*100
            out=np.sqrt(self._rolling(dd*dd,length,"mean"))
        elif name in {"aroonup", "aroondown"}:
            values=h if name=="aroonup" else l
            if n>=length:
                windows=np.lib.stride_tricks.sliding_window_view(values,length)
                reversed_windows=windows[:,::-1]
                pos=(length-1-np.argmax(reversed_windows,axis=1) if name=="aroonup"
                     else length-1-np.argmin(reversed_windows,axis=1))
                out[length-1:]=100*(pos+1)/length
        elif name in {"donchianupper", "pricechannelhigh"}:
            out=self._rolling(h,length,"max")
        elif name in {"donchianlower", "pricechannellow"}:
            out=self._rolling(l,length,"min")
        elif name in {"donchianmid", "pricechannelmid"}:
            out=(self._rolling(h,length,"max")+self._rolling(l,length,"min"))/2
        elif name == "keltnerchannel":
            out=self._ema(src,length)
        elif name in {"keltnerupper", "keltnerlower"}:
            mult=nums[1] if len(nums)>1 else 2.0; mid=self._ema(src,length); atr=self._wilder(tr,length)
            out=mid+(mult*atr if name=="keltnerupper" else -mult*atr)
        elif name in {"vwap", "volumeweightedaverageprice"}:
            tp=(h+l+c)/3
            # Intraday/session VWAP; daily reset follows each exchange-local date.
            dates=self.session_dates
            pv=tp*v; sums=pd.Series(pv).groupby(dates).cumsum().to_numpy(); vols=pd.Series(v).groupby(dates).cumsum().to_numpy()
            out=np.divide(sums,vols,out=np.full(n,np.nan),where=vols!=0)
        elif name in {"fisher", "fishertransform"}:
            hh=self._rolling(h,length,"max"); ll=self._rolling(l,length,"min")
            x=np.divide(2*(src-ll),hh-ll,out=np.zeros(n),where=hh!=ll)-1
            val=0.0; prev=0.0
            for j in range(n):
                val=.33*x[j]+.67*val if np.isfinite(x[j]) else val
                val=max(-.999,min(.999,val)); out[j]=.5*math.log((1+val)/(1-val))+.5*prev; prev=out[j]
        elif name in {"zscore", "standardscore"}:
            mean=self._rolling(src,length,"mean"); sd=self._rolling(src,length,"std")
            out=np.divide(src-mean,sd,out=out,where=sd!=0)
        elif name in {"linearregslope", "linearregression slope".replace(" ", "")}:
            x=np.arange(length,dtype=float); xc=x-x.mean(); den=np.sum(xc*xc)
            if n>=length: out[length-1:]=np.convolve(src,xc[::-1],mode="valid")/den
        elif name in {"linearregvalue", "timeseriesforecast", "linearregline"}:
            x=np.arange(length,dtype=float); xm=x.mean(); den=np.sum((x-xm)**2)
            if n>=length:
                means=self._rolling(src,length,"mean")
                slopes=np.convolve(src,(x-xm)[::-1],mode="valid")/den
                out[length-1:]=means[length-1:]+slopes*((length-1)-xm)
        elif name in {"typicalprice", "typical"}:
            out=(h+l+c)/3
        elif name in {"weightedclose", "weightedprice"}:
            out=(h+l+2*c)/4
        elif name in {"medianprice", "median"}:
            out=(h+l)/2
        elif name in {"awesomeoscillator", "ao"}:
            median=(h+l)/2; fast=ints[0] if ints else 5; slow=ints[1] if len(ints)>1 else 34
            out=self._rolling(median,fast,"mean")-self._rolling(median,slow,"mean")
        elif name in {"acceleratoroscillator", "ac"}:
            median=(h+l)/2; fast=ints[0] if ints else 5; slow=ints[1] if len(ints)>1 else 34
            ao=self._rolling(median,fast,"mean")-self._rolling(median,slow,"mean")
            smooth=ints[2] if len(ints)>2 else 5
            out=ao-self._rolling(ao,smooth,"mean")
        elif name in {"priceenvelopeupper", "movavgenvelopeupper"}:
            percent=nums[1] if len(nums)>1 else 2.5
            out=self._rolling(src,length,"mean")*(1+percent/100)
        elif name in {"priceenvelopelower", "movavgenvelopelower"}:
            percent=nums[1] if len(nums)>1 else 2.5
            out=self._rolling(src,length,"mean")*(1-percent/100)
        elif name in {"standarderror", "stderr"}:
            out=self._rolling(src,length,"std")/math.sqrt(length)
        elif name in {"standarderrorupper", "standarderrorlower"}:
            x=np.arange(length,dtype=float); xm=x.mean(); den=np.sum((x-xm)**2)
            if n>=length:
                means=self._rolling(src,length,"mean")
                slopes=np.convolve(src,(x-xm)[::-1],mode="valid")/den
                regression=means[length-1:]+slopes*((length-1)-xm)
                stderr=self._rolling(src,length,"std")/math.sqrt(length)
                mult=nums[1] if len(nums)>1 else 2.0
                out[length-1:]=regression+(mult*stderr[length-1:] if name=="standarderrorupper" else -mult*stderr[length-1:])
        elif name in {"doji", "insidebar", "outsidebar", "bullengulfing", "bearengulfing", "bullharami", "bearharami", "hammer", "hangingman", "shootingstar", "morningstar", "eveningstar", "morningdoji", "eveningdoji", "piercingline", "darkcloudcover", "threewhitesoldiers", "threeblackcrows", "keyreversalup", "keyreversaldown", "gapup", "gapdown"}:
            body=np.abs(c-o); bar_range=h-l; upper=h-np.maximum(c,o); lower=np.minimum(c,o)-l
            po=np.r_[np.nan,o[:-1]]; pc=np.r_[np.nan,c[:-1]]
            ph=np.r_[np.nan,h[:-1]]; pl=np.r_[np.nan,l[:-1]]
            previous_range=np.r_[np.nan,bar_range[:-1]]
            prev_bull=pc>po; prev_bear=pc<po; current_bull=c>o; current_bear=c<o
            if name=="doji": signal=body<=.1*bar_range
            elif name=="insidebar": signal=(h<ph)&(l>pl)
            elif name=="outsidebar": signal=(h>ph)&(l<pl)
            elif name=="bullengulfing": signal=(pc<po)&(c>o)&(o<=pc)&(c>=po)
            elif name=="bearengulfing": signal=(pc>po)&(c<o)&(o>=pc)&(c<=po)
            elif name=="bullharami": signal=prev_bear&current_bull&(o>=pc)&(c<=po)&(body<np.abs(pc-po))
            elif name=="bearharami": signal=prev_bull&current_bear&(o<=pc)&(c>=po)&(body<np.abs(pc-po))
            elif name in {"hammer","hangingman"}: signal=(lower>=2*body)&(upper<=body)&(current_bull if name=="hammer" else current_bear)
            elif name=="shootingstar": signal=(upper>=2*body)&(lower<=body)&(c<o)
            elif name=="gapup": signal=l>ph
            elif name=="gapdown": signal=h<pl
            elif name in {"keyreversalup","keyreversaldown"}:
                signal=(l<pl)&(c>ph) if name=="keyreversalup" else (h>ph)&(c<pl)
            elif name=="piercingline": signal=prev_bear&current_bull&(o<pl)&(c>(po+pc)/2)&(c<po)
            elif name=="darkcloudcover": signal=prev_bull&current_bear&(o>ph)&(c<(po+pc)/2)&(c>po)
            elif name in {"threewhitesoldiers","threeblackcrows"}:
                p2o=np.r_[np.nan,np.nan,o[:-2]]; p2c=np.r_[np.nan,np.nan,c[:-2]]
                p2bull=p2c>p2o; p2bear=p2c<p2o
                if name=="threewhitesoldiers": signal=p2bull&prev_bull&current_bull&(pc>p2c)&(c>pc)&(o<=pc)&(o>=po)
                else: signal=p2bear&prev_bear&current_bear&(pc<p2c)&(c<pc)&(o>=pc)&(o<=po)
            elif name in {"morningstar","morningdoji"}:
                p2o=np.r_[np.nan,np.nan,o[:-2]]; p2c=np.r_[np.nan,np.nan,c[:-2]]
                middle_small=np.abs(pc-po)<np.abs(p2c-p2o)*.5
                if name=="morningdoji": middle_small=middle_small&(np.abs(pc-po)<=.1*previous_range)
                signal=(p2c<p2o)&middle_small&current_bull&(c>(p2o+p2c)/2)
            else:
                p2o=np.r_[np.nan,np.nan,o[:-2]]; p2c=np.r_[np.nan,np.nan,c[:-2]]
                middle_small=np.abs(pc-po)<np.abs(p2c-p2o)*.5
                if name=="eveningdoji": middle_small=middle_small&(np.abs(pc-po)<=.1*previous_range)
                signal=(p2c>p2o)&middle_small&current_bear&(c<(p2o+p2c)/2)
            out=np.asarray(signal,dtype=np.float64)
        elif name in {"percentchange", "pctchange", "percentincrease", "percentdecrease"}:
            lag=np.r_[np.full(length,np.nan),src[:-length]] if length<n else np.full(n,np.nan)
            change=np.divide(src-lag,lag,out=out,where=lag!=0)*100
            out=np.abs(change) if name=="percentincrease" else -np.abs(change) if name=="percentdecrease" else change
        elif name in {"averageclosegreateropen", "avgclosegreateropen", "avgcloselessthanavgopen", "avgclosegreaterthanavgopen", "pricegreaterthanaverage", "pricelessthanaverage"}:
            close_avg=self._rolling(c,length,"mean"); open_avg=self._rolling(o,length,"mean")
            if name in {"avgclosegreateropen", "avgclosegreaterthanavgopen"}: signal=close_avg>open_avg
            elif name=="avgcloselessthanavgopen": signal=close_avg<open_avg
            elif name=="pricegreaterthanaverage": signal=src>self._rolling(src,length,"mean")
            elif name=="pricelessthanaverage": signal=src<self._rolling(src,length,"mean")
            else: signal=close_avg>open_avg
            out=np.asarray(signal,dtype=np.float64)
        elif name in {"bidaskspread", "spread", "spread_diff"}:
            bid=self.fields.get("bid",np.full(n,np.nan)); ask=self.fields.get("ask",np.full(n,np.nan))
            out=ask-bid
        elif name in {"bidaskmidpoint", "quoteprice"}:
            bid=self.fields.get("bid",np.full(n,np.nan)); ask=self.fields.get("ask",np.full(n,np.nan))
            out=(ask+bid)/2
        elif name in {"bidaskvolratio", "bidaskvolumeratio"}:
            bids=self.fields.get("bid_size",np.full(n,np.nan)); asks=self.fields.get("ask_size",np.full(n,np.nan))
            out=np.divide(bids,asks,out=out,where=asks!=0)
        elif name in {"marketdepthbidprice", "marketdepthaskprice", "marketdepthbidsize", "marketdepthasksize"}:
            level=max(1,int(nums[0])) if nums else 1
            prefix={"marketdepthbidprice":"mbo_bid_price_", "marketdepthaskprice":"mbo_ask_price_",
                    "marketdepthbidsize":"mbo_bid_size_", "marketdepthasksize":"mbo_ask_size_"}[name]
            out=self.fields.get(prefix+str(level),out)
        elif name in {"marketdepthbid", "marketdepthask", "marketdepthimbalance", "marketdepthspread", "marketdepthmicroprice"}:
            levels=max(1,int(nums[0])) if nums else 5
            if name in {"marketdepthbid", "marketdepthask"}:
                prefix="mbo_bid_size_" if name=="marketdepthbid" else "mbo_ask_size_"
                columns=[self.fields.get(prefix+str(k),np.full(n,np.nan)) for k in range(1,levels+1)]
                out=np.nansum(np.vstack(columns),axis=0)
                valid=np.isfinite(np.vstack(columns)).any(axis=0); out[~valid]=np.nan
            elif name=="marketdepthimbalance":
                bid=np.vstack([self.fields.get(f"mbo_bid_size_{k}",np.full(n,np.nan)) for k in range(1,levels+1)])
                ask=np.vstack([self.fields.get(f"mbo_ask_size_{k}",np.full(n,np.nan)) for k in range(1,levels+1)])
                b=np.nansum(bid,axis=0); a=np.nansum(ask,axis=0); total=b+a
                out=np.divide(b-a,total,out=out,where=total!=0)
                out[~(np.isfinite(bid).any(axis=0)&np.isfinite(ask).any(axis=0))]=np.nan
            elif name=="marketdepthspread": out=self.fields.get("mbo_spread",out)
            else: out=self.fields.get("mbo_microprice",out)
        elif name in {"marketdepthbboimbalance", "marketdepthtopimbalance"}:
            bid=self.fields.get("mbo_bid_size_1",np.full(n,np.nan)); ask=self.fields.get("mbo_ask_size_1",np.full(n,np.nan))
            out=np.divide(bid-ask,bid+ask,out=out,where=(bid+ask)!=0)
        elif name in {"marketdepthweightedmid", "marketdepthweightedmidprice"}:
            bp=self.fields.get("mbo_bid_price_1",np.full(n,np.nan)); ap=self.fields.get("mbo_ask_price_1",np.full(n,np.nan))
            bs=self.fields.get("mbo_bid_size_1",np.full(n,np.nan)); ass=self.fields.get("mbo_ask_size_1",np.full(n,np.nan))
            out=np.divide(ap*bs+bp*ass,bs+ass,out=out,where=(bs+ass)!=0)
        elif name in {"marketfacilitationindex", "marketfacilitation", "mfi_bar"}:
            out=np.divide(h-l,v,out=out,where=v!=0)
        elif name in {"volumeaverage", "volumeavg", "volume_avg"}:
            out=self._rolling(v,length,"mean")
        elif name in {"volumeratio", "volume_ratio"}:
            avg=self._rolling(v,length,"mean"); out=np.divide(v,avg,out=out,where=avg!=0)
        elif name in {"volumebreakout", "volume_breakout"}:
            avg=self._rolling(v,length,"mean"); mult=nums[1] if len(nums)>1 else 1.5
            out=np.asarray(v>avg*mult,dtype=np.float64)
        elif name in {"volumegradient", "volume_gradient"}:
            out=np.diff(v,prepend=np.nan)
        elif name in {"marneyvolumeindicator", "marneyvolume", "marneyrangeindicator", "marneyrange"}:
            # Time-of-day profile over prior sessions. MultiCharts describes
            # these as hourly historical profiles; hour is the profile key.
            stamps=pd.DatetimeIndex(self.bars.timestamp)
            if stamps.tz is not None: stamps=stamps.tz_convert(self.session_timezone)
            slot=np.asarray(stamps.hour,dtype=np.int16); keys=self.session_dates
            bars=pd.DataFrame({"date":keys,"slot":slot,"high":h,"low":l,"close":c,"volume":v})
            daily=bars.groupby(["date","slot"],sort=False).agg(
                high=("high","max"),low=("low","min"),close=("close","last"),volume=("volume","sum")
            ).reset_index()
            if name in {"marneyvolumeindicator","marneyvolume"}:
                daily["value"]=daily.volume
            else:
                prior_close=daily.groupby("date",sort=False).close.shift(1)
                tr_hour=np.maximum(daily.high-daily.low,np.maximum(np.abs(daily.high-prior_close),np.abs(daily.low-prior_close)))
                tr_hour=tr_hour.where(prior_close.notna(),daily.high-daily.low)
                daily["value"]=tr_hour
            daily["profile"]=daily.groupby("slot",sort=False).value.transform(
                lambda x: x.shift(1).rolling(length,min_periods=1).mean())
            mapping={(int(row.date),int(row.slot)):float(row.profile) for row in daily.itertuples()}
            out=np.fromiter((mapping.get((int(d),int(s)),np.nan) for d,s in zip(keys,slot)),dtype=np.float64,count=n)
        elif name in {"volumeup", "volume_up", "volumedown", "volume_down"}:
            direction=np.diff(c,prepend=np.nan)
            out=np.where(direction>0,v,0) if name in {"volumeup","volume_up"} else np.where(direction<0,v,0)
        elif name in {"consecutiveups", "consecutiveup", "consecutivedowns", "consecutivedown"}:
            direction=np.diff(c,prepend=np.nan); want_up=name in {"consecutiveups","consecutiveup"}
            counts=np.zeros(n,dtype=np.float64)
            for j in range(1,n):
                hit=direction[j]>0 if want_up else direction[j]<0
                counts[j]=counts[j-1]+1 if hit else 0
            out=counts
        elif name in {"volumetoday", "volume_today", "volumecumulative"}:
            out=pd.Series(v).groupby(self.session_dates,sort=False).cumsum().to_numpy()
        elif name in {"dayopen", "dayhigh", "daylow", "firstbarofday", "firstbarofweek", "firstbarofmonth", "firstbarofyear"}:
            if name=="dayopen": out=self._grouped("open",self.session_dates,"first")
            elif name=="dayhigh": out=self._grouped("high",self.session_dates,"max",cumulative=True)
            elif name=="daylow": out=self._grouped("low",self.session_dates,"min",cumulative=True)
            else:
                keymap={"firstbarofday":self.session_dates,"firstbarofweek":self.week_keys,
                        "firstbarofmonth":self.month_keys,"firstbarofyear":self.year_keys}
                keys=keymap[name]; out=np.asarray(pd.Series(keys).ne(pd.Series(keys).shift(1)),dtype=np.float64)
        elif name in {"newhighforchart", "newlowforchart", "newhighforday", "newlowforday", "newhighforweek", "newlowforweek", "newhighformonth", "newlowformonth", "newhighforyear", "newlowforyear", "newintradayhigh", "newintradaylow", "newintradayhilo"}:
            intraday=name.startswith("newintraday")
            if intraday: keys=self.session_dates
            elif name.endswith("forday"): keys=self.session_dates
            elif name.endswith("forweek"): keys=self.week_keys
            elif name.endswith("formonth"): keys=self.month_keys
            elif name.endswith("foryear"): keys=self.year_keys
            else: keys=None
            is_high=name.startswith("newhigh") or name=="newintradayhigh"
            if intraday and name=="newintradayhilo":
                maxes=self._grouped("high",keys,"max",cumulative=True)
                mins=self._grouped("low",keys,"min",cumulative=True)
                out=np.asarray((h>=maxes)|(l<=mins),dtype=np.float64)
            else:
                source=h if is_high else l
                if keys is None:
                    extrema=np.maximum.accumulate(source) if is_high else np.minimum.accumulate(source)
                else:
                    extrema=self._grouped("high" if is_high else "low",keys,"max" if is_high else "min",cumulative=True)
                out=np.asarray(source>=extrema if is_high else source<=extrema,dtype=np.float64)
        elif name in {"newndayhigh", "newndaylow"}:
            source=h if name=="newndayhigh" else l
            out=self._rolling(source,length,"max" if name=="newndayhigh" else "min")
        elif name in {"breakoutofxbarhigh", "breakoutofxbarlow"}:
            is_high=name.endswith("high"); source=h if is_high else l
            prev=self._rolling(source,length,"max" if is_high else "min")
            prev=np.r_[np.nan,prev[:-1]]
            out=np.asarray(c>prev if is_high else c<prev,dtype=np.float64)
        elif name in {"floortraderpivot", "pivotpoint", "floortraderpivots", "yesterdayhigh", "yesterdaylow", "yesterdayclose"}:
            ph=self._prior_daily("high"); pl=self._prior_daily("low"); pc=self._prior_daily("close")
            pivot=(ph+pl+pc)/3
            if name=="yesterdayhigh": out=ph
            elif name=="yesterdaylow": out=pl
            elif name=="yesterdayclose": out=pc
            elif name in {"floortraderpivot","pivotpoint","floortraderpivots"}: out=pivot
            elif name=="floortraderresistance1": out=2*pivot-pl
            elif name=="floortradersupport1": out=2*pivot-ph
        elif name in {"floortraderresistance1", "floortradersupport1"}:
            ph=self._prior_daily("high"); pl=self._prior_daily("low"); pc=self._prior_daily("close")
            pivot=(ph+pl+pc)/3
            out=2*pivot-pl if name=="floortraderresistance1" else 2*pivot-ph
        elif name in {"nextthirdfriday", "expiration"}:
            stamps=pd.DatetimeIndex(self.bars.timestamp)
            if stamps.tz is not None: stamps=stamps.tz_convert(self.session_timezone)
            def third_friday(year,month):
                first=pd.Timestamp(year=year,month=month,day=1)
                return first+pd.Timedelta(days=(4-first.weekday())%7+14)
            values=np.empty(n,dtype=np.float64)
            month_keys=np.asarray(stamps.year*100+stamps.month)
            positions=pd.Series(np.arange(n)).groupby(month_keys,sort=False)
            for month_key,indexes in positions:
                idx=indexes.to_numpy(dtype=np.int64); first_stamp=stamps[int(idx[0])]
                year,month=int(first_stamp.year),int(first_stamp.month); current=third_friday(year,month)
                following_month=month+1; following_year=year
                if following_month>12: following_month=1; following_year+=1
                following=third_friday(following_year,following_month)
                current_date=np.asarray(stamps[idx].strftime("%Y%m%d"),dtype=np.int64)
                values[idx]=np.where(current_date<=int(current.strftime("%Y%m%d")),
                                     int(current.strftime("%Y%m%d")),int(following.strftime("%Y%m%d")))
            out=values
        elif name in {"accumdistbuypr", "accumdistprvol", "moneyflow"}:
            mfm=np.divide((c-l)-(h-c),h-l,out=np.zeros(n),where=h!=l)
            out=mfm*v if name=="accumdistbuypr" else np.cumsum(mfm*v)
        elif name in {"volatility", "volatilitystddev"}:
            returns=np.r_[np.nan,np.diff(np.log(np.maximum(src,1e-300)))]
            out=self._rolling(returns,length,"std")*100
        elif name in {"correlation", "correlationcoefficient"}:
            second=args[1] if len(args)>1 else None
            other,_=self._src(second)
            if second is None and len(self.fields.get("close_data2",()))==n:
                other=self.fields["close_data2"]
            mx=self._rolling(src,length,"mean"); my=self._rolling(other,length,"mean")
            vx=self._rolling(src*src,length,"mean")-mx*mx
            vy=self._rolling(other*other,length,"mean")-my*my
            cov=self._rolling(src*other,length,"mean")-mx*my
            den=np.sqrt(np.maximum(vx,0)*np.maximum(vy,0))
            out=np.divide(cov,den,out=out,where=den>0)
        elif name in {"spreadratio", "spread_diff", "spreaddiff"}:
            # MultiCharts Spread studies compare two explicitly supplied data
            # series; missing Data2 is NaN instead of silently substituting a
            # different instrument or the bid/ask quote.
            second=args[1] if len(args)>1 else None
            other, _ = self._src(second)
            if second is None and len(self.fields.get("close_data2",()))==n:
                other=self.fields["close_data2"]
            if name=="spreadratio":
                out=np.divide(src,other,out=out,where=other!=0)
            else:
                out=src-other
        elif name in {"kama", "adaptivemovingaverage"}:
            fast=ints[1] if len(ints)>1 else 2; slow=ints[2] if len(ints)>2 else 30
            change=np.full(n,np.nan)
            if length<n: change[length:]=np.abs(src[length:]-src[:-length])
            volatility=self._rolling(np.r_[np.nan,np.abs(np.diff(src))],length,"sum")
            er=np.divide(change,volatility,out=np.zeros(n),where=np.isfinite(volatility)&(volatility!=0))
            fast_sc=2/(fast+1); slow_sc=2/(slow+1); sc=(er*(fast_sc-slow_sc)+slow_sc)**2
            if n:
                out[0]=src[0]
                for j in range(1,n): out[j]=out[j-1]+(sc[j] if np.isfinite(sc[j]) else slow_sc**2)*(src[j]-out[j-1])
        elif name in {"parabolicsar", "psar", "parabolic"}:
            step=nums[0] if nums else 0.02; maximum=nums[1] if len(nums)>1 else 0.2
            if n>1:
                bull=c[1]>=c[0]; ep=h[0] if bull else l[0]; af=step; out[0]=l[0] if bull else h[0]
                for j in range(1,n):
                    prior=out[j-1]
                    sar=prior+af*(ep-prior)
                    if bull:
                        sar=min(sar,l[j-1],l[j-2] if j>1 else l[j-1])
                        if l[j]<sar:
                            bull=False; sar=ep; ep=l[j]; af=step
                        elif h[j]>ep: ep=h[j]; af=min(maximum,af+step)
                    else:
                        sar=max(sar,h[j-1],h[j-2] if j>1 else h[j-1])
                        if h[j]>sar:
                            bull=True; sar=ep; ep=h[j]; af=step
                        elif l[j]<ep: ep=l[j]; af=min(maximum,af+step)
                    out[j]=sar
        elif name in {"stochrsi", "stochasticrsi"}:
            rsi=self._rsi(src,length); lookback=ints[1] if len(ints)>1 else length
            lo=self._rolling(rsi,lookback,"min"); hi=self._rolling(rsi,lookback,"max")
            out=np.divide(100*(rsi-lo),hi-lo,out=out,where=hi!=lo)
        elif name in {"aroonoscillator", "aroonosc"}:
            up=self.series("aroonup",length); down=self.series("aroondown",length); out=up-down
        elif name in {"choppiness", "choppinessindex", "chop"}:
            trsum=self._rolling(tr,length,"sum"); hh=self._rolling(h,length,"max"); ll=self._rolling(l,length,"min")
            ratio=np.divide(trsum,hh-ll,out=out.copy(),where=hh!=ll)
            out=100*np.log10(ratio)/math.log10(length) if length>1 else out
        elif name in {"dpo", "detrendedpriceoscillator"}:
            shift=max(0,length//2+1); base=self._rolling(src,length,"mean")
            if shift==0: out=src-base
            elif shift<n: out[shift:]=src[:-shift]-base[shift:]
        elif name in {"easeofmovement", "emv"}:
            midpoint=(h+l)/2; distance=np.diff(midpoint,prepend=midpoint[0]); box=np.divide(v,h-l,out=np.zeros(n),where=h!=l)
            raw=np.divide(distance,box,out=np.zeros(n),where=box!=0)
            out=self._rolling(raw,length,"mean")
        elif name in {"forceindex", "force"}:
            raw=np.diff(c,prepend=c[0])*v; out=self._ema(raw,length)
        elif name in {"elderbullpower", "bullpower"}:
            out=h-self._ema(src,length)
        elif name in {"elderbearpower", "bearpower"}:
            out=l-self._ema(src,length)
        elif name in {"ichimokuconversion", "ichimokconversion", "tenkansen"}:
            out=(self._rolling(h,length,"max")+self._rolling(l,length,"min"))/2
        elif name in {"ichimokubase", "kijunsen"}:
            base=ints[1] if len(ints)>1 else 26
            out=(self._rolling(h,base,"max")+self._rolling(l,base,"min"))/2
        elif name in {"ichimokuspana", "senkouspana"}:
            base=ints[1] if len(ints)>1 else 26
            conv=(self._rolling(h,length,"max")+self._rolling(l,length,"min"))/2
            kijun=(self._rolling(h,base,"max")+self._rolling(l,base,"min"))/2
            out=(conv+kijun)/2
        elif name in {"ichimokuspanb", "senkouspanb"}:
            span=ints[1] if len(ints)>1 else 52
            out=(self._rolling(h,span,"max")+self._rolling(l,span,"min"))/2
        elif name in {"relativevigorindex", "rvi"}:
            num=self._rolling(c-o,length,"mean"); den=self._rolling(h-l,length,"mean")
            out=np.divide(num,den,out=out,where=den!=0)
        elif name in {"swingindex", "accumswingindex", "accumulativeswingindex"}:
            # Wilder Swing Index, using only current OHLC and prior O/C.  LimitMove
            # is configurable (default 50); SI stays within +/-100 by construction.
            limit_move=nums[0] if nums else 50.0
            pc=np.r_[np.nan,c[:-1]]; po=np.r_[np.nan,o[:-1]]
            a=np.abs(h-pc); b=np.abs(l-pc); cc=np.abs(h-l)
            q=.25*np.abs(pc-po)
            r=np.where((a>=b)&(a>=cc),a-.5*b+q,
                       np.where((b>=cc),b-.5*a+q,cc+q))
            k=np.maximum(a,b)
            si=np.divide(50*((c-pc)+.5*(c-o)+.25*(pc-o)),
                         r*max(abs(limit_move),1e-12),out=np.full(n,np.nan),where=r!=0)
            si=np.clip(si,-100,100)
            out=np.cumsum(np.nan_to_num(si)) if name in {"accumswingindex","accumulativeswingindex"} else si
        elif name in {"zigzagpercent", "zigzagpct", "zigzagpoints", "zigzagpoint"}:
            # Causal ZigZag: publish a swing only once price reverses by the
            # configured threshold. Earlier candidate extremes are never
            # backfilled, so strategy use cannot see a future-confirmed pivot.
            threshold=abs(nums[0]) if nums else (5.0 if name in {"zigzagpercent","zigzagpct"} else 1.0)
            source=src; direction=0; extreme=float(source[0]) if n else 0.0
            for j in range(1,n):
                px=float(source[j])
                if direction>=0:
                    if px>=extreme: extreme=px
                    reversal=(extreme-px)/max(abs(extreme),1e-12)*100 if name in {"zigzagpercent","zigzagpct"} else extreme-px
                    if reversal>=threshold:
                        out[j]=extreme; direction=-1; extreme=px
                if direction<=0:
                    if px<=extreme: extreme=px
                    reversal=(px-extreme)/max(abs(extreme),1e-12)*100 if name in {"zigzagpercent","zigzagpct"} else px-extreme
                    if reversal>=threshold:
                        out[j]=extreme; direction=1; extreme=px
        elif name in {"momentumincreasing", "momentumdecreasing"}:
            mom=src-np.r_[np.full(length,np.nan),src[:-length]] if length<n else np.full(n,np.nan)
            prior=np.r_[np.nan,mom[:-1]]
            out=np.asarray(mom>prior if name=="momentumincreasing" else mom<prior,dtype=np.float64)
        elif name in {"cciaverage", "ccisma"}:
            tp=(h+l+c)/3
            mean=self._rolling(tp,length,"mean")
            dev=self._rolling(np.abs(tp-mean),length,"mean")
            cci=np.divide(tp-mean,.015*dev,out=np.full(n,np.nan),where=dev!=0)
            smooth=ints[1] if len(ints)>1 else length
            out=self._rolling(cci,smooth,"mean")
        elif name in {"islandreversalup", "islandreversaldown"}:
            ph=np.r_[np.nan,h[:-1]]; pl=np.r_[np.nan,l[:-1]]
            ph2=np.r_[np.nan,np.nan,h[:-2]]; pl2=np.r_[np.nan,np.nan,l[:-2]]
            # One-bar island: the prior bar gaps away from its neighbors, and
            # the current bar gaps back across that isolated bar.
            out=np.asarray(((ph<pl2)&(l>ph)) if name=="islandreversalup" else
                           ((pl>ph2)&(h<pl)),dtype=np.float64)
        elif name in {"alligatorjaw", "alligatorteeth", "alligatorlips", "chaosalligatorjaw", "chaosalligatorteeth", "chaosalligatorlips", "gatorupper", "gatorlower", "chaosgatorupper", "chaosgatorlower"}:
            price_input=src_value if src_value is not None else _PLMedian()
            alligator_source,_=self._src(price_input)
            jaw_len=ints[0] if ints else 7; teeth_len=ints[1] if len(ints)>1 else 8; lips_len=ints[2] if len(ints)>2 else 5
            jaw=self._wilder(alligator_source,jaw_len)
            teeth=self._wilder(alligator_source,teeth_len)
            lips=self._wilder(alligator_source,lips_len)
            if name in {"alligatorjaw","chaosalligatorjaw"}: out=jaw
            elif name in {"alligatorteeth","chaosalligatorteeth"}: out=teeth
            elif name in {"alligatorlips","chaosalligatorlips"}: out=lips
            elif name in {"gatorupper","chaosgatorupper"}: out=np.abs(jaw-teeth)
            elif name in {"gatorlower","chaosgatorlower"}: out=-np.abs(teeth-lips)
        elif name in {"chaosfractalhigh", "chaosfractallow", "fractals"}:
            strength=max(2,ints[0] if ints else 2); source=h if name!="chaosfractallow" else l
            for j in range(2*strength,n):
                p=j-strength; window=source[p-strength:p+strength+1]
                is_extreme=source[p]==(np.max(window) if name!="chaosfractallow" else np.min(window))
                if is_extreme and not np.any(window[strength+1:]==source[p]): out[j]=source[p]
        else:
            raise KeyError(name)
        self.cache[cache_key] = np.asarray(out,dtype=np.float64)
        return self.cache[cache_key]

    def value(self, name, index, *args):
        canonical=INDICATOR_ALIASES.get(name.lower(),name.lower())
        if canonical in {"spreadratio","spread_diff","spreaddiff"} and len(args)>=2:
            first_values,first_field=self._src(args[0])
            second_values,second_field=self._src(args[1])
            if second_field.endswith("_data2") and second_field not in self.fields:
                return np.nan
            first=float(first_values[index]); second=float(second_values[index])
            if canonical=="spreadratio":
                return first/second if second else np.nan
            return first-second
        values=self.series(name,*args)
        return float(values[index]) if 0 <= int(index) < len(values) else np.nan


INDICATOR_ALIASES = {
    "simplemovingaverage":"sma", "movavgsimple":"sma", "movingaverage":"sma",
    "exponentialmovingaverage":"ema", "movavgexponential":"ema",
    "weightedaverage":"wma", "weightedmovingaverage":"wma", "movavgweighted":"wma",
    "triangularmovingaverage":"trima", "movavgtriangular":"trima",
    "hullmovingaverage":"hma", "movavghull":"hma", "averagetruerange":"atr",
    "relative strength index":"rsi", "relativestrengthindex":"rsi",
    "commoditychannelindex":"cci", "commchannelindex":"cci",
    "averagedirectionalindex":"adx", "standarddev":"stddev", "stdev":"stddev",
    "rateofchange":"roc", "onbalancevolume":"obv", "accumulationdistribution":"accumdist",
    "moneyflowindex":"mfi", "pricevolumetrend":"pvt", "ultimateoscillator":"ultosc",
    "stochasticfastk":"fastk", "stochasticfastd":"fastd", "stochasticslowk":"slowk",
    "stochasticslowd":"slowd", "percentb":"bollingerpercentb", "bollingerbandupper":"bollingerupper",
    "bollingerbandlower":"bollingerlower", "priceosc":"priceoscillator", "mass":"massindex",
    "volumeweightedaverageprice":"vwap", "fishertransform":"fisher",
    "waverage":"wma", "wildersaverage":"wilderaverage", "exponentialaverage":"ema",
    "adaptivemovingaverage":"kama", "stochasticrsi":"stochrsi", "aroonosc":"aroonoscillator",
    "choppinessindex":"choppiness", "detrendedpriceoscillator":"dpo", "emv":"easeofmovement",
    "force":"forceindex", "bullpower":"elderbullpower", "bearpower":"elderbearpower",
    "tenkansen":"ichimokconversion", "kijunsen":"ichimokubase", "senkouspana":"ichimokuspana",
    "senkouspanb":"ichimokuspanb", "rvi":"relativevigorindex", "psar":"parabolicsar",
    "ichimokuconversion":"ichimokconversion",
    "typical":"typicalprice", "weightedprice":"weightedclose", "median":"medianprice",
    "ao":"awesomeoscillator", "ac":"acceleratoroscillator", "stderr":"standarderror",
    "movavgenvelopeupper":"priceenvelopeupper", "movavgenvelopelower":"priceenvelopelower",
    "pctchange":"percentchange", "%chg":"percentchange", "percentchg":"percentchange",
    "avgclosegreaterthanavgopen":"averageclosegreateropen",
    "averageclosegreaterthanaverageopen":"averageclosegreateropen",
    "averagecloselessthanaverageopen":"avgcloselessthanavgopen",
    "bidaskvolumeratio":"bidaskvolratio", "marketfacilitation":"marketfacilitationindex",
    "volavg":"volumeaverage", "volume_ratio":"volumeratio", "volume_gradient":"volumegradient",
    "volume_today":"volumetoday", "volume_up":"volumeup", "volume_down":"volumedown",
    "marketdepthbidvolume":"marketdepthbid", "marketdepthaskvolume":"marketdepthask",
    "marketdepthdepthimbalance":"marketdepthimbalance",
    "marketdepthbboimbalance":"marketdepthbboimbalance", "marketdepthtopimbalance":"marketdepthtopimbalance",
    "marketdepthweightedmid":"marketdepthweightedmid", "marketdepthweightedmidprice":"marketdepthweightedmidprice",
    "bidandaskvolratio":"bidaskvolratio", "bidaskvolumeratio":"bidaskvolratio",
    "marketfacilitation":"marketfacilitationindex", "volavg":"volumeaverage",
    "volume_avg":"volumeaverage", "volume_ratio":"volumeratio", "volume_gradient":"volumegradient",
    "volume_today":"volumetoday", "volume_up":"volumeup", "volume_down":"volumedown",
    "newhighforday":"newhighforday", "newlowforday":"newlowforday",
    "floortraderpivots":"floortraderpivot", "pivotpoint":"floortraderpivot",
    "yesterdayhigh":"yesterdayhigh", "yesterdaylow":"yesterdaylow", "yesterdayclose":"yesterdayclose",
    "accumdistbuypr":"accumdistbuypr", "accumdistprvol":"accumdistprvol",
    "accumswingindex":"accumswingindex", "accumulativeswingindex":"accumswingindex",
    "swingindex":"swingindex", "pivothigh":"pivothigh", "pivotlow":"pivotlow",
    "zigzag%":"zigzagpercent", "zigzagpct":"zigzagpercent", "zigzagpoint":"zigzagpoints",
    "momentumincreasing":"momentumincreasing", "momentumdecreasing":"momentumdecreasing",
    "ccisma":"cciaverage", "islandreversalup":"islandreversalup", "islandreversaldown":"islandreversaldown",
    "spreadratio":"spreadratio", "spreaddiff":"spreaddiff", "spread_diff":"spreaddiff",
    "alligatorjaw":"alligatorjaw", "alligatorteeth":"alligatorteeth", "alligatorlips":"alligatorlips",
    "chaosalligatorjaw":"chaosalligatorjaw", "chaosalligatorteeth":"chaosalligatorteeth",
    "chaosalligatorlips":"chaosalligatorlips", "gatorupper":"gatorupper", "gatorlower":"gatorlower",
    "chaosgatorupper":"chaosgatorupper", "chaosgatorlower":"chaosgatorlower",
    "chaosfractalhigh":"chaosfractalhigh", "chaosfractallow":"chaosfractallow",
    "newintradayhigh":"newintradayhigh", "newintradaylow":"newintradaylow", "newintradayhilo":"newintradayhilo",
    "marneyvolume":"marneyvolume", "marneyvolumeindicator":"marneyvolumeindicator",
    "marneyrange":"marneyrange", "marneyrangeindicator":"marneyrangeindicator",
}


INDICATOR_DEFINITIONS = {
    "SMA": "rolling arithmetic mean", "EMA": "exponential moving average, alpha=2/(N+1)",
    "WMA": "linearly weighted moving average", "TRIMA": "two-pass triangular moving average",
    "HMA": "Hull moving average: WMA(2*WMA(N/2)-WMA(N), sqrt(N))",
    "TrueRange": "max(high-low, abs(high-prior close), abs(low-prior close))",
    "ATR": "Wilder-smoothed True Range", "RSI": "Wilder relative strength index",
    "CCI": "(typical price - N-period mean) / (0.015 * mean absolute deviation)",
    "ADX": "Wilder-smoothed directional movement index", "PlusDI/MinusDI": "directional movement divided by Wilder ATR, in percent",
    "ADXR": "mean of current ADX and ADX N bars ago", "MACD": "EMA(fast)-EMA(slow)",
    "MACDSignal": "EMA of MACD line", "MACDHistogram": "MACD line minus signal line",
    "BollingerUpper/Lower": "SMA plus/minus deviations times population standard deviation",
    "FastK": "100*(close-lowest low)/(highest high-lowest low)", "SlowK/SlowD": "smoothed stochastic %K/%D",
    "WilliamsR": "-100*(highest high-close)/(highest high-lowest low)",
    "OBV": "cumulative volume signed by close-to-close direction", "AccumDist": "cumulative money-flow volume",
    "ChaikinOsc": "fast EMA minus slow EMA of accumulation/distribution", "MFI": "volume-weighted RSI of typical price",
    "VolumeOscillator": "percent difference between fast and slow volume EMAs", "PVT": "cumulative volume times fractional price change",
    "TRIX": "one-period percentage change of triple EMA", "PPO": "percentage difference between fast and slow EMAs",
    "UltimateOscillator": "weighted average buying-pressure ratios over three lookbacks",
    "UlcerIndex": "root mean square of percentage drawdowns from rolling highs",
    "AroonUp/Down": "percent of lookback elapsed since rolling high/low", "Donchian": "rolling high, low, or midpoint",
    "Keltner": "EMA centerline plus/minus ATR multiple", "VWAP": "session cumulative typical-price-volume / volume",
    "Fisher": "bounded-price Fisher transform with recursive smoothing", "ZScore": "(value - rolling mean)/rolling standard deviation",
    "LinearRegSlope/Value": "least-squares rolling regression slope/current endpoint forecast",
    "Correlation": "rolling Pearson correlation between two input series",
    "KAMA": "Kaufman adaptive moving average using efficiency ratio and squared smoothing constant",
    "ParabolicSAR": "iterative stop-and-reverse trend state with acceleration factor",
    "StochRSI": "RSI position within its rolling RSI high-low range, scaled 0-100",
    "AroonOscillator": "AroonUp minus AroonDown", "Choppiness": "100*log10(sum(TrueRange)/range)/log10(N)",
    "DPO": "price shifted by half-period minus the rolling mean", "EaseOfMovement": "midpoint change adjusted by range and volume",
    "ForceIndex": "EMA of price change times volume", "ElderBullPower": "high minus EMA",
    "ElderBearPower": "low minus EMA", "IchimokuConversion/Base/SpanA/SpanB": "midpoints of rolling high-low ranges; spans here are unshifted to avoid future projection",
    "RelativeVigorIndex": "rolling mean(close-open) divided by rolling mean(high-low)",
}
