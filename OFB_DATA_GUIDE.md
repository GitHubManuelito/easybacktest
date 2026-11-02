# OrderFlowBridge data bundle

The data is installed under `data/OFB/`. This directory is local-only and ignored by Git because the original bundle is about 9.1 GB. The OHLC file has one identical duplicate in the source folder; only one copy is stored here.

| File | Contents | Reader support |
| --- | --- | --- |
| `ofb_bars_NQ.csv` | 1-minute NQ OHLC, volume, UpVol, DownVol; timestamps `MM/DD/YYYY` + `HH:MM` | Full OHLCV loading; timestamps are localized as `America/New_York`; UpVol/DownVol become `flow_buy_volume`, `flow_sell_volume`, `flow_delta`, `flow_delta_pct`, and `flow_imbalance` |
| `ofb_hist.bin` | Timestamp-keyed 80-value volume-profile/order-flow feature records | Selective slot reading for POC, VAH/VAL, prior POC, VWAP and standard deviation, depth totals, walls, and largest trade; arbitrary `OFB_GetAt(slot, key)` slots are read when the slot is literal in the script |
| `ofb_depth.bin` | Timestamp-keyed 64-level depth feed (128 float values per record) | Kept intact; raw ladder rendering/recursive depth APIs are not yet mapped because the vendor's value layout is not fully described in this export |
| `ofb_gex.bin` | Timestamp-keyed gamma/option feature records; header advertises 128 strikes, record has 273 float values | Literal `OFB_GexScalar(slot, key)` slots can be loaded; strike-rank heatmap/`OFB_GexAt` decoding is not enabled until the packed row layout is verified |
| `ofb_cot.bin` | 1,518 release-time rows × 247 named COT series | Named series can be loaded from string literals in `OFB_Cot` / `OFB_CotValid`; values are joined as-of release time to avoid look-ahead |
| `ofb_no_data_days.json` | Date-level gaps, including no-GEX dates | Preserved as coverage metadata |

The timestamped binary keys use `YYYYMMDDHHMMSS`, interpreted in the same chart timezone as the OFB bar CSV, then converted to UTC for joins. The COT reader uses the latest release key at or before each bar. Historical profile/depth/GEX slots only join on matching timestamps; missing rows remain NaN and are never forward-filled.

## Use the feed

Run the engine and choose NQ, then provide the dataset folder `data/OFB` (or use `--data data/OFB`). The engine finds `ofb_bars_NQ.csv` in that folder. Ordinary NQ runs continue to use the existing longer NQ history unless this source is selected explicitly.

PowerLanguage code can call the supported Bridge helper names such as `OFB_POC(2)`, `OFB_VAH(2)`, `OFB_VAL(2)`, `OFB_PriorPOC(2)`, `OFB_VWAP(2)`, `OFB_VWAP_SD(2)`, `OFB_BidDepth(50)`, `OFB_AskDepth(50)`, `OFB_BidWallPx(1)`, `OFB_AskWallSz(1)`, `OFB_BigTrade()`, and `OFB_BigTradePx()`. Literal `OFB_GetAt(slot, key)` and `OFB_GexScalar(slot, key)` calls can access slots present in the supplied feed at the current chart-bar timestamp; arbitrary historical key lookups are not simulated. The regular PowerLanguage parser still does not load external DLLs: remove `DefineDLLFunc` declarations and DLL loader calls from a script and use the engine's built-in equivalents.

Raw depth and GEX files remain in the local data folder even where their vendor-specific multidimensional APIs are not yet decoded. The guide deliberately distinguishes a validated file container from fully verified meaning for each packed value.
