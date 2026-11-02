# MultiCharts pre-built study coverage

This page maps the official MultiCharts pre-built study catalog to what this engine can calculate from the data currently installed under `data/`. It describes **strategy-usable calculations**; the engine does not reproduce MultiCharts chart panels, colors, alert windows, drawing objects, or licensed add-ons. Indicator formulas and calls are listed in [`INDICATORS.md`](INDICATORS.md).

MultiCharts' official catalog contains more than 200 studies, including both indicators and strategy signal templates. A catalog entry is only marked supported here when the engine has a defined calculation or the relevant PowerLanguage order/runtime behavior. Similar names do not imply bit-for-bit identical initialization, session templates, or chart displacement.

## Available with the installed OHLCV bars

The NQ, ES, GC, and BTC primary histories are OHLCV bars. The following catalog groups therefore work for all four assets (subject to their specified warm-up and session assumptions):

| MultiCharts catalog studies | Engine support |
|---|---|
| % Decrease, % Increase, % Chg, Rate of Chg; Momentum, Momentum Increasing/Decreasing | Implemented as percent-change and momentum studies. |
| ADX, ADXR, DMI, +DI/-DI, ATR, Volatility, Volatility Std Dev | Implemented; DMI outputs are exposed as separate scalar studies. |
| Avg Close < Avg Open; Avg Close > Avg Open; Price < Avg; Price > Avg | Implemented as rolling comparisons, default length 14. |
| Accum Dist - BuyPr, Accum Dist - PrVol; Accum Swing Index; Swing Index; Money Flow | Implemented; Swing Index uses OHLC and prior O/C, CSI is cumulative. |
| Bollinger Bands; Breakout of X Bar High/Low; Price Channel; New N-Day HiLo; New High/Low for chart/day/week/month/year | Implemented as numeric levels or flags; breakout checks use prior completed bars. |
| Candlestick studies: 3 White Soldiers/3 Black Crows, Engulfing, Harami, Doji, Hammer/Hanging Man, Morning/Evening Doji/Star, Piercing/Dark Cloud, Shooting Star; Gap Up/Down; Inside/Outside Bar; Key Reversal | Implemented as numeric pattern flags using documented, explicit approximations. Island reversals are also supported. |
| CCI, CCI Average, Chaikin Osc, Chaos AC/Awesome, Chaos Alligator/Gator/Fractals | Implemented as scalar lines or flags. Alligator is Wilder-smoothed with the catalog's 7/8/5 default periods; chart forward shifts (8/5/3) are omitted for strategy causality. Fractals and pivots emit only after confirmation. |
| Consecutive Ups/Downs; Correlation; DPO; Ease of Movement; Expiration/Next 3rd Friday; First Bar of Day/Month/Week/Year | Implemented subject to timezone and Data2 assumptions. Correlation accepts two data series in the study call. |
| Floor Trader Pivots; OHLC Yesterday; Day Open/High/Low and intraday Hi/Lo; Typical Price; Weighted Close | Implemented. Session-local calculations reset at the selected timezone's date boundary; no exchange calendar is applied. |
| Ichimoku Cloud; Keltner Channel; Linear Reg Curve/Line; MACD and MACD gradient components; Mass Index; Market Facilitation Index; Mov Avg Adaptive, Envelope, Exponential, Hull, Triangular, Weighted | Implemented as separate numeric study functions. Ichimoku spans and Alligator lines are not forward-plotted. |
| Marney Range Indicator; Marney Volume Indicator | Implemented as previous-session hourly profiles (defaults to 10 prior sessions); current session is excluded from its reference average. |
| OBV; Parabolic SAR; Pivot High/Low; Price Osc; Price Volume Trend; RSI; Stochastic Fast/Slow; TRIX; Ulcer Index; Ultimate Oscillator; Volume Avg/Breakout/Down/Gradient/Oscillator/Rate of Change/Ratio/Today/Up; VWAP; ZigZag %/Points | Implemented. Pivot/fractal/ZigZag values are confirmation-only to prevent future leakage/repainting in strategy logic. |

Functions return a scalar current-bar value. Studies that are charted as several plots have separate named functions (for example, `BollingerUpper` and `BollingerLower`, or `AlligatorJaw`, `AlligatorTeeth`, and `AlligatorLips`). NumericRef output parameters are implemented for `VolumeProfileLevels` and `OFB_CotPercentile`; other arbitrary user-defined functions are not imported automatically. See `INDICATORS.md` for the supported call forms.

## Available only when the corresponding feed is selected

| Catalog feature | Data / coverage in this workspace | Runtime behavior |
|---|---|---|
| Bid & Ask; Bid & Ask Vol Ratio; Spread - Diff | TBBO has trades with contemporaneous best bid/ask and displayed sizes for NQ, MNQ, MES, and some other imported futures sessions. | Quote studies are enabled only if the selected dataset has the required quote columns. Coverage is far shorter than the four long OHLCV histories. |
| Spread - Ratio | Requires two supplied data series (typically Data1 and Data2). | `SpreadRatio(series1, series2)` and `SpreadDiff(series1, series2)` calculate the current value; Data2 must be available and synchronized. |
| Market Depth on Chart | NQ order-event data can be converted to book features for about 32 session dates; it is not a full-length depth history. MGC's purported full-depth files were invalid; the retained MGC source is a BBO proxy, not L2. | MarketDepth functions require synchronized NQ book-size fields. They fail with a clear data requirement when those fields are absent. Do not treat BBO as depth. |
| Block Trades per Bar | Trade-level feeds exist for some assets, but the imported standard cache is bar-aggregated and does not preserve every trade's size in every backtest path. | Not exposed as a historical study. MultiCharts itself documents its Block Trades per Bar study as real-time only and says it does not plot on historical data. |

## Not calculable from the installed synchronized backtest inputs

| Catalog entries / group | Why it is unavailable |
|---|---|
| Absolute Breadth; Advance-Decline Line/Ratio; Arms Index (TRIN); McClellan Oscillator; Market Thrust | These need synchronized advances/declines or breadth observations across a market universe. No such time series is installed. |
| Black-Scholes; Implied Volatility; Option Price/Delta/Gamma/Theta/Vega | Options prints/definitions are present in imported archives, but they do not overlap the local NQ minute bars (the archive's NQ option dates are later). A valid synchronized option chain/volatility surface is not supplied. |
| Open Interest | No open-interest series is included in the primary OHLCV or selected TBBO schemas. Volume is not a valid substitute. |
| Market-depth replay on ES/GC/BTC or full-period NQ | The available order-book coverage is incomplete and asset-limited. Best-bid/offer snapshots cannot reconstruct missing queue levels or historical depth. |
| HPI; third-party/proprietary indicators | HPI requires Open Interest, which is absent. Other proprietary items without reproducible formulas or required inputs are not approximated. |

## Chart-only studies and Auto Trading Signals

PaintBars, ShowMe studies, gradient fills, multi-line drawing studies, automatic trend lines, floating text, gapless chart rendering, and chart layout/display options are visual platform features. The engine can export strategy plots, but does not emulate those chart interactions. MultiCharts' Auto Trading Signals catalog entries are ready-made strategy templates (entries, stops, targets, exits), rather than independent market-data indicators. Equivalent logic can be expressed in a `.pl` strategy using the engine's supported PowerLanguage orders and stops; the named MultiCharts template files themselves are not bundled here.

## Data and calculation caveats

- OHLC bars cannot reveal intrabar tick order. The simulator must use its documented bar-fill policy for stops, targets, and same-bar conflicts.
- The imported TBBO feeds provide prints and best quotes for bounded date ranges; they do not extend the 16-year OHLCV coverage.
- NQ market-depth results are limited to the order-event dates. A request for a longer period cannot create missing depth observations.
- Session studies use the configured timezone and date grouping. MultiCharts session templates, holidays, and exchange-specific trading-day boundaries are not automatically inferred from OHLCV timestamps.
- `SwingIndex`, candle patterns, Island Reversal, pivots, fractals, Alligator, and ZigZag have explicit approximations documented in `INDICATORS.md`; compare outputs against MultiCharts before relying on identical values.

## Reference

The full vendor catalog is [MultiCharts Pre-built Studies](https://www.multicharts.com/trading-software/index.php?title=Pre-built_Studies_in_MultiCharts). It currently describes 200+ studies and includes both indicator studies and Auto Trading Signals. MultiCharts notes that Block Trades per Bar is real-time only in [its study description](https://www.multicharts.com/trading-software/index.php?title=Block_Trades_per_Bar).
