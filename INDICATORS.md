# PowerLanguage technical indicator reference

This engine now calculates the common bar-based indicators below from Data1 OHLCV. Calls return one numeric value for the current bar; assign the value to a variable or use it in a condition. Examples:

```PowerLanguage
emaValue = EMA(Close, 20);
atrValue = AvgTrueRange(14);
macdValue = MACD(Close, 12, 26);
upperBand = BollingerBand(Close, 20, 2);
rsiValue = RSI(Close, 14);
```

Periods must be positive integers. Indicator warm-up values are NaN until enough history exists (or until the recurrence has a valid seed). They therefore should not be used as signals before their warm-up completes. Indicator outputs do not change order timing or fills by themselves.

`VolumeProfileLevels(WindowBars, UseTickRows, TicksPerRow, NumRows, ValueAreaPct, oPOC, oVAH, oVAL)` is also available as a native multi-output function. It returns `1` when the rolling profile is ready and writes the three output variables; it returns `0` during warm-up. It follows the supplied template's evenly distributed bar-volume and greedy value-area method, capped at 500 rows. The current bar is included. Output-reference parameters are supported for this function and `OFB_CotPercentile`.

## Supported indicator families and definitions

| Indicator / supported names | Definition used by this engine |
|---|---|
| SMA, SimpleMovingAverage, Average, AverageFC | Arithmetic mean over the last N values. |
| EMA, XAverage, ExponentialMovingAverage | Exponential mean with alpha `2/(N+1)`, seeded from the first available value. |
| WMA, WAverage, WeightedAverage | Linear weights 1 through N, with the newest bar receiving weight N. |
| TRIMA, TriangularMovingAverage | Two successive simple averages with approximately triangular weighting. |
| HMA, HullMovingAverage | `WMA(2*WMA(price,N/2)-WMA(price,N),sqrt(N))`. |
| KAMA, AdaptiveMovingAverage | Kaufman adaptive average using N-bar efficiency ratio and squared fast/slow smoothing constants (defaults 2 and 30). |
| WilderAverage, WildersAverage | Wilder recursive smoothing, seeded by an N-value mean. |
| TrueRange, TR | Maximum of high-low, absolute high-prior-close, and absolute low-prior-close. |
| ATR, AvgTrueRange, AverageTrueRange | Wilder-smoothed True Range. |
| StdDev, StandardDev, StDev | Rolling standard deviation; the existing PowerLanguage `StdDev(value,N,dataType)` call supports population (`dataType=1`) or sample (`dataType=2`) convention. |
| Variance | Square of the rolling population standard deviation. |
| Momentum, Mom | Current source minus source N bars ago. |
| ROC, RateOfChange | Percentage change versus N bars ago. `VolumeRateOfChange` applies the same formula to volume. |
| RSI, RelativeStrengthIndex | Wilder relative-strength oscillator, scaled 0–100. Flat gain/loss windows return 50. |
| CCI, CommodityChannelIndex | Typical-price deviation from its rolling mean divided by `0.015 * mean absolute deviation`. |
| ADX, AverageDirectionalIndex | Wilder-smoothed directional movement strength, scaled 0–100. |
| PlusDI, DIPlus; MinusDI, DIMinus | Positive/negative directional movement divided by Wilder ATR, scaled 0–100. |
| ADXR | Mean of ADX now and ADX N bars earlier. |
| MACD, MACDLine | Fast EMA minus slow EMA (defaults 12 and 26 if periods are omitted). |
| MACDSignal, MACDTrigger | EMA of the MACD line; defaults 12, 26, and 9. |
| MACDHistogram, MACDHist | MACD line minus its signal EMA. |
| BollingerBand, BollingerUpper, BollingerLower | Rolling mean plus/minus the requested number of population standard deviations. A negative deviation multiplier on `BollingerBand` selects the lower band. |
| BollingerPercentB, PercentB | Source position within a 2-standard-deviation Bollinger envelope, where 0 is the lower band and 1 is the upper band. |
| BollingerBandwidth, Bandwidth | Band width divided by the rolling mean, expressed as a percentage. |
| FastK, StochasticFastK | `100*(close-lowestLow)/(highestHigh-lowestLow)`. |
| FastD, StochasticFastD | Moving average of FastK. |
| SlowK, StochasticSlowK | Smoothed FastK; default smoothing is 3. |
| SlowD, StochasticSlowD | Moving average of SlowK; default smoothing is 3. |
| WilliamsR, PercentR | `-100*(highestHigh-close)/(highestHigh-lowestLow)`, bounded from -100 to 0. |
| StochRSI, StochasticRSI | RSI's location within its rolling RSI high-low range, scaled 0–100. |
| OBV, OnBalanceVolume | Cumulative volume signed by the direction of close-to-close change. May be referenced as `OBV` or `OBV()`. |
| AccumDist, AccumulationDistribution | Cumulative volume multiplied by the close location value within each bar. |
| ChaikinOsc | Difference between fast and slow EMAs of the accumulation/distribution line (defaults 3 and 10). |
| MFI, MoneyFlowIndex | Positive versus negative rolling typical-price volume flow, scaled 0–100. |
| VolumeOscillator, VolOsc | Percentage difference between fast and slow volume EMAs (defaults 5 and 20). |
| PVT, PriceVolumeTrend | Cumulative volume multiplied by fractional close-to-close price change. |
| TRIX | One-bar percentage change in a triple-smoothed EMA. |
| PPO | Percentage difference between fast and slow price EMAs (defaults 12 and 26). |
| PriceOscillator | Difference between fast and slow price EMAs (defaults 10 and 20). |
| MassIndex | Rolling sum of the ratio of a 9-period EMA of high-low range to its second EMA. |
| UltimateOscillator, UltOsc | Weighted buying-pressure ratios over short, medium, and long periods (defaults 7, 14, 28), scaled 0–100. |
| UlcerIndex | Root mean square of percentage drawdown from the rolling high. |
| AroonUp, AroonDown | Percentage of the lookback represented by the most recent rolling high or low. |
| AroonOscillator | AroonUp minus AroonDown. |
| DonchianUpper/Lower/Mid; PriceChannelHigh/Low/Mid | Rolling high, rolling low, or their midpoint. |
| KeltnerUpper/Lower/Channel | EMA centerline plus/minus an ATR multiple; `KeltnerChannel` returns the EMA centerline. |
| VWAP, VolumeWeightedAveragePrice | Session-local cumulative typical-price volume divided by cumulative volume. VWAP resets at the selected session timezone's calendar date boundary. |
| Fisher, FisherTransform | Bounded rolling-price Fisher transform with recursive smoothing. |
| ZScore, StandardScore | `(source - rolling mean) / rolling standard deviation`. |
| LinearRegSlope | Least-squares slope over the last N values. |
| LinearRegValue, LinearRegLine, TimeSeriesForecast | Least-squares regression evaluated at the current window's last x-coordinate. |
| Correlation, CorrelationCoefficient | Rolling Pearson correlation between two supplied series and an N-bar period. |
| ParabolicSAR, PSAR, Parabolic | Iterative stop-and-reverse series with configurable acceleration step and maximum (defaults 0.02 and 0.2). |
| DPO, DetrendedPriceOscillator | Price shifted by roughly half the selected period minus the current rolling mean. |
| EaseOfMovement, EMV | Midpoint change adjusted for range and volume, then averaged over N bars. |
| ForceIndex, Force | EMA of close change multiplied by volume. |
| ElderBullPower / BullPower | High minus the selected price EMA. |
| ElderBearPower / BearPower | Low minus the selected price EMA. |
| IchimokuConversion / TenkanSen | Midpoint of the N-bar high-low range (commonly N=9). |
| IchimokuBase / KijunSen | Midpoint of the configured base-period high-low range (commonly 26). |
| IchimokuSpanA / SenkouSpanA | Midpoint between conversion and base lines, unshifted. |
| IchimokuSpanB / SenkouSpanB | Midpoint of the configured span-period high-low range, unshifted. |
| RelativeVigorIndex, RVI | Rolling mean of close-open divided by rolling mean of high-low. |
| TypicalPrice | `(high+low+close)/3`. |
| WeightedClose | `(high+low+2*close)/4`. |
| MedianPrice | `(high+low)/2`. |
| AwesomeOscillator, AO | Fast minus slow simple averages of median price (defaults 5 and 34). |
| AcceleratorOscillator, AC | Awesome Oscillator minus its short simple average (defaults 5 and 34 for AO, then 5). |
| PriceEnvelopeUpper/Lower | N-bar simple average adjusted by the requested percent above or below. |
| StandardError | Rolling population standard deviation divided by square root of N. |
| StandardErrorUpper/Lower | Rolling least-squares endpoint plus/minus a multiple of standard error (default multiple 2). |
| Doji | Numeric 1 when candle body is at most 10% of its range; otherwise 0. |
| InsideBar / OutsideBar | Numeric 1 when the current high-low range is inside / outside the prior bar's range. |
| BullEngulfing / BearEngulfing | Numeric 1 when the current body engulfs the prior opposite-color candle body. |
| Hammer / ShootingStar | Numeric 1 using a 2:1 long-shadow to body ratio with the opposite shadow no larger than the body. |
| MorningStar / EveningStar | Three-bar reversal approximation: long first body, small middle body, opposite-color final body closing through the first body's midpoint. |
| GapUp / GapDown | Numeric 1 when the current bar's entire range is above / below the prior bar. |
| SwingIndex | Wilder Swing Index from current OHLC and prior close; optional LimitMove defaults to 50, output bounded to -100..100. |
| AccumSwingIndex / AccumulativeSwingIndex | Cumulative Swing Index, a running sum of SwingIndex. |
| PivotHigh / PivotLow | Confirmed pivot price emitted on the confirmation bar. Call `PivotHigh(High, leftBars, rightBars)` / `PivotLow(Low, leftBars, rightBars)`; defaults are 2/2. This delayed output avoids future leakage. |
| ZigZagPercent / ZigZagPoints | Causal ZigZag swing values, emitted only after the requested reversal confirms; default thresholds are 5% and 1 point. No backfilled/repainting pivot values. |
| CCI_Average | Rolling average of CCI; `CCIAverage(price, cciLength, smoothingLength)`, both periods default to 14. |
| MomentumIncreasing / MomentumDecreasing | Numeric 1 if the N-bar momentum is higher/lower than the previous bar's momentum. |
| IslandReversalUp / IslandReversalDown | One-bar gap-island reversal approximation, confirmed by a gap back across the isolated bar. |
| AlligatorJaw / AlligatorTeeth / AlligatorLips | Median-price Wilder-smoothed lines with common 13/8/5 periods. MultiCharts' forward chart displacements are omitted to avoid look-ahead. |
| GatorUpper / GatorLower | Absolute jaw-teeth / teeth-lips distance; lower is negative. Uses unshifted causal Alligator lines. |
| ChaosFractalHigh / ChaosFractalLow | Pivot high/low with 2 bars on either side by default; output appears only after right-side confirmation. |
| NewIntradayHigh / NewIntradayLow / NewIntradayHiLo | Current session's cumulative high/low break flags. Session boundaries use the selected timezone's calendar date, not an exchange session template. |
| SpreadRatio / SpreadDiff | Ratio/difference of two supplied series, e.g. `SpreadRatio(Close, Close of Data2)`; depends on Data2 and its alignment. |
| MarketDepthBBOImbalance / MarketDepthTopImbalance | `(best bid size - best ask size)/(best bid size + best ask size)`, requiring true NQ L2/L3 top-of-book sizes. |
| MarketDepthWeightedMidPrice | Size-weighted midpoint `(ask price*bid size + bid price*ask size)/(bid size+ask size)`, requiring synchronized NQ L2/L3 top level. |
| MarneyRange / MarneyVolume | Average hourly True Range / volume for the same timestamp-hour across prior sessions (default 10); the current session is excluded. This is a bar-data approximation of the vendor's intraday profile behavior. |

## Platform and calculation boundaries

This is a defined subset of MultiCharts PowerLanguage, not a clone of every built-in study or the platform runtime. See [`MULTICHARTS_STUDY_SUPPORT.md`](MULTICHARTS_STUDY_SUPPORT.md) for the full catalog cross-reference and dataset limits. Some studies need exchange breadth, options, market depth, tick history, proprietary components, or multiple output/reference parameters. Those are not interchangeable with OHLCV formulas. In particular, the classic multi-output `Stochastic(...)` output-reference form, market breadth studies, option-pricing studies, chart drawing/alerts, and licensed third-party studies are not represented by this single-value OHLCV interface. Use the named FastK/FastD/SlowK/SlowD functions above for stochastic values.

The formulas above are explicit engine definitions and may differ from a MultiCharts study's exact initialization, session template, data stream, default parameters, or intrabar update mode. `VWAP` and `SessionEndTime` use timestamp/session approximations rather than a loaded MultiCharts session template. Bars cannot reproduce tick-by-tick IOG, exact historical order of highs/lows, or bid/ask event sequencing. For independent reference, see MultiCharts' [pre-built studies catalog](https://www.multicharts.com/trading-software/index.php?title=Pre-built_Studies_in_MultiCharts) and [PowerLanguage documentation](https://www.multicharts.com/trading-software/index.php?title=Category:PowerLanguage).
