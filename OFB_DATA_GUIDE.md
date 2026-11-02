# OrderFlowBridge data bundle

The data is installed under `data/OFB/`. This directory is local-only and ignored by Git because the original bundle is about 9.1 GB. The OHLC file has one identical duplicate in the source folder; only one copy is stored here.

| File | Contents | Reader support |
| --- | --- | --- |
| `ofb_bars_NQ.csv` | 1-minute NQ OHLC, volume, UpVol, DownVol; timestamps `MM/DD/YYYY` + `HH:MM` | Full OHLCV loading; timestamps are localized as `America/New_York`; UpVol/DownVol become `flow_buy_volume`, `flow_sell_volume`, `flow_delta`, `flow_delta_pct`, and `flow_imbalance` |
| `ofb_hist.bin` | Timestamp-keyed 80-value volume-profile/order-flow feature records | Selective slot reading for POC, VAH/VAL, prior POC, VWAP and standard deviation, depth totals, walls, and largest trade; arbitrary `OFB_GetAt(slot, key)` slots are read when the slot is literal in the script |
| `ofb_depth.bin` | Timestamp-keyed 64-level depth feed (128 packed float64 words per record) | `OFB_DepthAt(side, level, what, key)` reads 64 depth levels per side; side 0=bid, 1=ask; what 0=price, 1=size. Each word packs two float32 values, laid out as bid price/size then ask price/size for each level |
| `ofb_gex.bin` | Timestamp-keyed gamma/option feature records; 17 scalar values and up to 128 packed strikes per row | `OFB_GexScalar(slot, key)`, `OFB_GexAt(rank, what, key)`, and `OFB_GexStrikes` are supported. Strike fields are price, gamma, call OI, and put OI; `what` is 0–3 |
| `ofb_cot.bin` | 1,518 release-time rows × 247 named COT series | Named series can be loaded from string literals in `OFB_Cot` / `OFB_CotValid`; values are joined as-of release time to avoid look-ahead |
| `ofb_no_data_days.json` | Date-level gaps, including no-GEX dates | Preserved as coverage metadata |

The timestamped binary keys use `YYYYMMDDHHMMSS`, interpreted in the same chart timezone as the OFB bar CSV, then converted to UTC for joins. The COT reader uses the latest release key at or before each bar. Historical profile/GEX scalar features join on matching timestamps; missing rows remain NaN and are never forward-filled. The depth and GEX strike ladders are read lazily from memory-mapped files, so the multi-gigabyte depth file is not expanded into a wide in-memory table.

## Use the feed

Run the engine and choose NQ, then provide the dataset folder `data/OFB` (or use `--data data/OFB`). The engine finds `ofb_bars_NQ.csv` in that folder. Ordinary NQ runs continue to use the existing longer NQ history unless this source is selected explicitly.

PowerLanguage can use the Bridge functions directly: `OFB_POC`, `OFB_VAH`, `OFB_VAL`, `OFB_PriorPOC`, `OFB_VWAP`, `OFB_VWAP_SD`, depth totals/walls, largest-trade values, COT lookups/rank, GEX scalar values and strike rows, and the raw `OFB_GetAt`, `OFB_GexAt`, and `OFB_DepthAt` calls. The supported loader/count/status/version functions are also provided natively. `DefineDLLFunc` declarations are accepted and ignored; the engine reads the local data files instead of loading the Windows DLL. Calls with an explicit historical key are resolved from the file's sorted timestamp index, including COT as-of release lookups.

The raw depth and GEX files remain local because of their size. Native readers implement the documented depth and strike APIs; drawing operations in supplied indicators are ignored by the backtest engine because they have no effect on orders or fills.
