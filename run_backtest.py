from __future__ import annotations

import argparse
import json
import math
import re
import sys
import textwrap
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd
from powerlanguage import PowerLanguageError, compile_program, parse_source, run_powerlanguage
from technical_indicators import INDICATOR_ALIASES
from ofb_data import attach_ofb_features

ROOT = Path(__file__).resolve().parent


def _round_report_values(value):
    """Round reported statistics to three decimals without changing booleans."""
    if isinstance(value, dict):
        return {key: _round_report_values(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_round_report_values(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_round_report_values(item) for item in value)
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (float, np.floating)):
        return round(float(value), 3)
    if isinstance(value, np.integer):
        return int(value)
    return value


def _report_text(value):
    if value is None:
        return "n/a"
    if isinstance(value, (bool, np.bool_)):
        return "Yes" if value else "No"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):,.3f}"
    if isinstance(value, (int, np.integer)):
        return f"{int(value):,}"
    if isinstance(value, (dict, list, tuple)):
        value = json.dumps(value, ensure_ascii=True, default=str, separators=(",", ":"))
    return textwrap.shorten(str(value).replace("\n", " "), width=110, placeholder="...")


def _write_pdf_report(out, asset, args, input_bars, all_source_bars, strategy_bars, inputs, stats,
                      chart_names):
    """Write a multi-page PDF report with rounded metrics and backtest charts."""
    pdf_path = out / "report.pdf"
    excluded = {"monte_carlo", "crypto_prop", "lucidflex", "lucidpro", "buy_and_hold_benchmark"}
    sections = [("Performance", {k: v for k, v in stats.items() if k not in excluded})]
    for key, title in (("buy_and_hold_benchmark", "Buy and Hold Benchmark"),
                       ("monte_carlo", "Monte Carlo Simulation"),
                       ("lucidflex", "LucidFlex Evaluation and Payout"),
                       ("lucidpro", "LucidPro Evaluation and Payout")):
        if key in stats:
            sections.append((title, stats[key]))
    if "crypto_prop" in stats:
        crypto_prop = stats["crypto_prop"]
        sections.append(("Crypto Prop Average Rules", {
            key: value for key, value in crypto_prop.items() if key != "accounts"
        }))
        for account_size, account_stats in crypto_prop.get("accounts", {}).items():
            sections.append((f"Crypto Prop Account - ${int(account_size):,}", account_stats))

    with PdfPages(pdf_path) as pdf:
        page_number = 0
        for title, metrics in sections:
            rows = list(metrics.items())
            # Split long sections so every row remains legible.
            for start in range(0, max(1, len(rows)), 28):
                page_number += 1
                fig, ax = plt.subplots(figsize=(8.27, 11.69))
                ax.axis("off")
                if page_number == 1:
                    fig.suptitle(f"{asset} Backtest Report", x=0.08, y=0.965, ha="left",
                                 fontsize=20, fontweight="bold", color="#183153")
                    metadata = (f"Strategy: {args.strategy.name}    |    Timeframe: {args.bar_timeframe}\n"
                                f"Data window: {args.data_period}    |    Source bars: {input_bars:,} / {all_source_bars:,}\n"
                                f"Strategy bars: {strategy_bars:,}    |    Inputs: {_report_text(inputs)}")
                    fig.text(0.08, 0.91, metadata, ha="left", va="top", fontsize=8.5, color="#46566b")
                    title_y = 0.82
                    table_bottom = 0.08
                    table_height = 0.68
                else:
                    fig.suptitle(f"{asset} Backtest Report", x=0.08, y=0.965, ha="left",
                                 fontsize=17, fontweight="bold", color="#183153")
                    title_y = 0.91
                    table_bottom = 0.08
                    table_height = 0.78
                ax.text(0.02, title_y, title, transform=ax.transAxes, ha="left", va="bottom",
                        fontsize=14, fontweight="bold", color="#276678")
                page_rows = rows[start:start + 28]
                cell_text = [[str(key).replace("_", " ").title(), _report_text(value)]
                             for key, value in page_rows]
                table = ax.table(cellText=cell_text, colLabels=["Statistic", "Value"],
                                 colWidths=[0.43, 0.57], cellLoc="left", colLoc="left",
                                 bbox=[0.02, table_bottom, 0.96, table_height])
                table.auto_set_font_size(False)
                table.set_fontsize(8.2 if len(page_rows) < 20 else 7.3)
                for (row, col), cell in table.get_celld().items():
                    cell.set_edgecolor("#d6dee8")
                    if row == 0:
                        cell.set_facecolor("#183153")
                        cell.get_text().set_color("white")
                        cell.get_text().set_fontweight("bold")
                    else:
                        cell.set_facecolor("#f3f6fa" if row % 2 == 0 else "white")
                        cell.get_text().set_color("#1c2838")
                fig.text(0.93, 0.025, f"Page {page_number}", ha="right", fontsize=8, color="#68778a")
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)

        for chart_name in chart_names:
            chart_path = out / chart_name
            if not chart_path.is_file():
                continue
            page_number += 1
            fig, ax = plt.subplots(figsize=(11.69, 8.27))
            ax.axis("off")
            ax.set_title(chart_name.replace("_", " ").replace(".png", "").title(),
                         loc="left", fontsize=16, fontweight="bold", color="#183153", pad=16)
            ax.imshow(plt.imread(chart_path))
            fig.text(0.94, 0.025, f"Page {page_number}", ha="right", fontsize=8, color="#68778a")
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
    return pdf_path


def _format_currency_axis(axis, which="y"):
    formatter = FuncFormatter(lambda value, _pos: f"-${abs(value):,.0f}" if value < 0 else f"${value:,.0f}")
    (axis.yaxis if which == "y" else axis.xaxis).set_major_formatter(formatter)


def _estimate_runtime(asset: str, strategy_name: str, timeframe: str, bars: int, preparation_seconds: float):
    """Estimate full run duration from completed backtests, preferring matching strategies."""
    asset_root = ROOT / "results" / asset
    records = []
    strategy_key = re.sub(r"[^A-Za-z0-9._-]+", "_", strategy_name).strip("._").lower()
    if asset_root.is_dir():
        for path in asset_root.glob("*/*/stats.json"):
            try:
                stats = json.loads(path.read_text(encoding="utf-8"))
                previous_bars = int(stats.get("bars", 0))
                if previous_bars <= 0:
                    continue
                previous_timeframe = str(stats.get("strategy_timeframe", ""))
                if previous_timeframe and previous_timeframe.lower() != timeframe.lower():
                    continue
                strategy_folder = path.parents[1].name.lower()
                exact_strategy = strategy_folder == strategy_key or strategy_folder.startswith(strategy_key + "_")
                if stats.get("processing_runtime_seconds") is not None:
                    seconds = float(stats["processing_runtime_seconds"])
                    prep_included = False
                else:
                    seconds = float(stats.get("runtime_seconds") or 0)
                    prep_included = True
                    if seconds <= 0:
                        run_tag = path.parent.name
                        if run_tag.startswith("discord-"):
                            try:
                                started = datetime.strptime(
                                    run_tag, "discord-%Y%m%d-%H%M%S"
                                ).replace(tzinfo=timezone.utc).timestamp()
                                seconds = max(0.0, path.stat().st_mtime - started)
                            except ValueError:
                                seconds = 0.0
                if seconds > 0:
                    records.append((3 if exact_strategy else 2, path.stat().st_mtime,
                                    seconds / previous_bars, prep_included))
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                continue
    if records:
        best_rank = max(record[0] for record in records)
        selected = sorted((record for record in records if record[0] == best_rank), key=lambda row: row[1], reverse=True)[:7]
        per_bar = float(np.median([row[2] for row in selected]))
        prep_included = any(row[3] for row in selected)
        estimated_processing = max(15.0, per_bar * max(1, bars))
        if prep_included:
            estimated_total = max(preparation_seconds + 15.0, estimated_processing)
            basis = f"{len(selected)} previous {asset} {timeframe} backtest(s)"
        else:
            estimated_total = preparation_seconds + estimated_processing
            basis = f"{len(selected)} previous runs of this strategy"
    else:
        estimated_processing = max(20.0, max(1, bars) * 0.01)
        estimated_total = preparation_seconds + estimated_processing
        basis = "rough first-run estimate; it will be refined from live progress"
    remaining = max(15.0, estimated_total - preparation_seconds)
    return estimated_total, remaining, basis


def _duration_text(seconds: float) -> str:
    value = max(0, int(round(seconds)))
    hours, remainder = divmod(value, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes}m"
    if minutes:
        return f"{minutes}m {seconds}s"
    return f"{seconds}s"


def _mbo_event_bars(source: Path, timeframe: str = "1min", include_depth: bool = False) -> pd.DataFrame:
    """Summarize London Strategic Edge order events without retaining raw ticks."""
    files = sorted(source.glob("*.csv.gz")) if source.is_dir() else [source]
    if source.is_dir():
        canonical_names = {file.name.replace(".OLD.csv.gz", ".csv.gz") for file in files}
        files = [file for file in files if ".OLD.csv.gz" not in file.name or file.name.replace(".OLD.csv.gz", ".csv.gz") not in canonical_names]
    if not files:
        raise ValueError(f"no compressed daily event CSVs found in {source}")
    if source.is_dir() and len(source.parents) >= 3:
        cache_name = "nq_order_events_1m_v3_depth.parquet" if include_depth else "nq_order_events_1m_v3.parquet"
        cache_path = source.parents[2] / "ohlcv" / cache_name
    else:
        cache_path = source.with_suffix(".mbo.1min.parquet")
    source_mtime = max((f.stat().st_mtime for f in files), default=0)
    if cache_path.exists() and cache_path.stat().st_mtime >= source_mtime:
        cached = pd.read_parquet(cache_path)
        if not include_depth or {"bid", "ask", "mbo_depth_imbalance_5"}.issubset(cached.columns):
            return cached
    parts = []
    book_orders = {}
    bid_levels, ask_levels = {}, {}
    snapshots = {}
    last_book_bucket = None
    last_trade_price = None

    def change_level(side_name, price, delta):
        if price <= 0 or side_name not in {"Buy", "Sell"}:
            return
        levels = bid_levels if side_name == "Buy" else ask_levels
        updated = levels.get(price, 0.0) + delta
        if updated <= 1e-9:
            levels.pop(price, None)
        else:
            levels[price] = updated

    def save_book(bucket):
        bid_prices = sorted(bid_levels, reverse=True)[:10]
        ask_prices = sorted(ask_levels)[:10]
        snapshot = {}
        for level in range(1, 11):
            bp = bid_prices[level - 1] if len(bid_prices) >= level else np.nan
            ap = ask_prices[level - 1] if len(ask_prices) >= level else np.nan
            bs = bid_levels.get(bp, np.nan) if np.isfinite(bp) else np.nan
            ass = ask_levels.get(ap, np.nan) if np.isfinite(ap) else np.nan
            snapshot[f"mbo_bid_price_{level}"] = bp
            snapshot[f"mbo_bid_size_{level}"] = bs
            snapshot[f"mbo_ask_price_{level}"] = ap
            snapshot[f"mbo_ask_size_{level}"] = ass
        bid, ask = snapshot["mbo_bid_price_1"], snapshot["mbo_ask_price_1"]
        bid_size, ask_size = snapshot["mbo_bid_size_1"], snapshot["mbo_ask_size_1"]
        snapshot["bid"] = bid
        snapshot["ask"] = ask
        snapshot["bid_size"] = bid_size
        snapshot["ask_size"] = ask_size
        snapshot["mbo_spread"] = ask - bid if np.isfinite(bid) and np.isfinite(ask) else np.nan
        bid_depth = sum(bid_levels[p] for p in bid_prices[:5])
        ask_depth = sum(ask_levels[p] for p in ask_prices[:5])
        snapshot["mbo_bid_depth_5"] = bid_depth
        snapshot["mbo_ask_depth_5"] = ask_depth
        total = bid_depth + ask_depth
        snapshot["mbo_depth_imbalance_5"] = (bid_depth - ask_depth) / total if total else 0.0
        if np.isfinite(bid) and np.isfinite(ask) and np.isfinite(bid_size) and np.isfinite(ask_size) and bid_size + ask_size:
            snapshot["mbo_microprice"] = (ask * bid_size + bid * ask_size) / (bid_size + ask_size)
        else:
            snapshot["mbo_microprice"] = np.nan
        snapshots[bucket] = snapshot

    size_cols = {"Add": "mbo_add_size", "Cancel": "mbo_cancel_size", "Modify": "mbo_modify_size",
                 "Fill": "mbo_fill_size", "Trade": "mbo_trade_size"}
    count_cols = {action: col.replace("_size", "_count") for action, col in size_cols.items()}
    for file_no, file in enumerate(files, start=1):
        print(f"Order-event conversion: {file_no}/{len(files)} {file.name}", file=sys.stderr, flush=True)
        for chunk in pd.read_csv(file, usecols=["timestamp", "action", "side", "price", "size", "order_id"],
                                 chunksize=250_000, compression="gzip" if file.suffix.lower() == ".gz" else None,
                                 low_memory=False):
            chunk["event_ts"] = pd.to_datetime(chunk.timestamp, utc=True, errors="coerce")
            chunk["timestamp"] = chunk["event_ts"].dt.floor(timeframe)
            chunk["size"] = pd.to_numeric(chunk["size"], errors="coerce").fillna(0.0)
            chunk["price"] = pd.to_numeric(chunk["price"], errors="coerce")
            chunk["action"] = chunk.action.astype("string").str.title()
            chunk["side"] = chunk.side.astype("string").str.title()
            chunk = chunk.sort_values("event_ts", kind="stable")
            if include_depth:
                for event in chunk.itertuples(index=False):
                    bucket = event.timestamp
                    if last_book_bucket is not None and bucket != last_book_bucket:
                        save_book(last_book_bucket)
                    last_book_bucket = bucket
                    action = str(event.action)
                    side_name = str(event.side)
                    order_id = str(event.order_id)
                    price = float(event.price) if pd.notna(event.price) else 0.0
                    size = float(event.size) if pd.notna(event.size) else 0.0
                    if action == "Reset":
                        book_orders.clear()
                        bid_levels.clear()
                        ask_levels.clear()
                    elif order_id not in {"", "0", "nan", "None"}:
                        old = book_orders.get(order_id)
                        if action == "Add":
                            if old is not None:
                                change_level(old[0], old[1], -old[2])
                            if side_name in {"Buy", "Sell"} and price > 0 and size > 0:
                                book_orders[order_id] = (side_name, price, size)
                                change_level(side_name, price, size)
                        elif action == "Modify":
                            if old is not None:
                                change_level(old[0], old[1], -old[2])
                            resolved_side = side_name if side_name in {"Buy", "Sell"} else (old[0] if old else "")
                            if resolved_side in {"Buy", "Sell"} and price > 0 and size > 0:
                                book_orders[order_id] = (resolved_side, price, size)
                                change_level(resolved_side, price, size)
                            else:
                                book_orders.pop(order_id, None)
                        elif action in {"Cancel", "Fill"} and old is not None:
                            removed = min(old[2], size) if size > 0 else old[2]
                            change_level(old[0], old[1], -removed)
                            remaining = old[2] - removed
                            if remaining > 1e-9:
                                book_orders[order_id] = (old[0], old[1], remaining)
                            else:
                                book_orders.pop(order_id, None)
            chunk = chunk.loc[chunk.timestamp.notna()]
            if chunk.empty:
                continue
            ev = pd.DataFrame(index=pd.Index(chunk.timestamp.unique(), name="timestamp")).sort_index()
            for action in size_cols:
                selected = chunk.action.eq(action)
                ev[count_cols[action]] = selected.groupby(chunk.timestamp).sum().reindex(ev.index, fill_value=0)
                ev[size_cols[action]] = chunk["size"].where(selected, 0).groupby(chunk.timestamp).sum().reindex(ev.index, fill_value=0)
                for side_name in ("Buy", "Sell"):
                    sided = selected & chunk.side.astype("string").str.title().eq(side_name)
                    stem = f"mbo_{action.lower()}_{side_name.lower()}"
                    ev[f"{stem}_count"] = sided.groupby(chunk.timestamp).sum().reindex(ev.index, fill_value=0)
                    ev[f"{stem}_size"] = chunk["size"].where(sided, 0).groupby(chunk.timestamp).sum().reindex(ev.index, fill_value=0)
            trades = chunk.loc[chunk.action.eq("Trade") & chunk.price.notna()]
            if trades.empty:
                bars = pd.DataFrame(index=ev.index, columns=["open", "high", "low", "close"], dtype=float)
                bars["volume"] = 0.0
                bars["trade_count"] = 0
                bars["mbo_buy_trade_volume"] = 0.0
                bars["mbo_sell_trade_volume"] = 0.0
                bars["up_ticks"] = 0
                bars["down_ticks"] = 0
            else:
                bars = trades.groupby("timestamp", sort=True).agg(
                    open=("price", "first"), high=("price", "max"), low=("price", "min"),
                    close=("price", "last"), volume=("size", "sum"), trade_count=("price", "count"),
                )
                for side, col in (("Buy", "mbo_buy_trade_volume"), ("Sell", "mbo_sell_trade_volume")):
                    bars[col] = trades.loc[trades.side.eq(side)].groupby("timestamp")["size"].sum().reindex(bars.index, fill_value=0)
                ordered_trades = trades.sort_values("event_ts", kind="stable")
                ordered_prices = ordered_trades.price.astype(float)
                changes = ordered_prices.diff()
                if last_trade_price is not None and len(changes):
                    changes.iloc[0] = ordered_prices.iloc[0] - last_trade_price
                bars["up_ticks"] = (changes > 0).groupby(ordered_trades.timestamp).sum().reindex(bars.index, fill_value=0)
                bars["down_ticks"] = (changes < 0).groupby(ordered_trades.timestamp).sum().reindex(bars.index, fill_value=0)
                last_trade_price = float(ordered_prices.iloc[-1])
            bars = bars.join(ev, how="outer")
            zero_fields = [column for column in bars.columns if column not in {"open", "high", "low", "close"}]
            bars[zero_fields] = bars[zero_fields].fillna(0)
            bars["flow_buy_volume"] = bars["mbo_buy_trade_volume"]
            bars["flow_sell_volume"] = bars["mbo_sell_trade_volume"]
            bars["flow_delta"] = bars["flow_buy_volume"] - bars["flow_sell_volume"]
            bars["flow_delta_pct"] = np.divide(bars["flow_delta"], bars["volume"],
                out=np.zeros(len(bars), dtype=float), where=bars["volume"].to_numpy(dtype=float) != 0) * 100.0
            bars["flow_imbalance"] = np.divide(bars["flow_buy_volume"] - bars["flow_sell_volume"],
                bars["flow_buy_volume"] + bars["flow_sell_volume"],
                out=np.zeros(len(bars), dtype=float),
                where=(bars["flow_buy_volume"] + bars["flow_sell_volume"]).to_numpy(dtype=float) != 0)
            bars["mbo_add_imbalance"] = bars["mbo_add_buy_size"] - bars["mbo_add_sell_size"]
            bars["mbo_cancel_imbalance"] = bars["mbo_cancel_buy_size"] - bars["mbo_cancel_sell_size"]
            bars["mbo_net_add_imbalance"] = bars["mbo_add_imbalance"] - bars["mbo_cancel_imbalance"]
            parts.append(bars)
    if include_depth and last_book_bucket is not None:
        save_book(last_book_bucket)
    if not parts:
        raise ValueError(f"no Trade events with usable prices found in {source}")
    raw = pd.concat(parts).sort_index()
    sums = [c for c in raw.columns if c not in {"open", "high", "low", "close"}]
    frame = raw.groupby(level=0, sort=True).agg(
        {**{"open": "first", "high": "max", "low": "min", "close": "last"}, **{c: "sum" for c in sums}}
    )
    frame["close"] = frame["close"].ffill()
    for field in ("open", "high", "low"):
        frame[field] = frame[field].fillna(frame["close"])
    frame = frame.loc[frame.close.notna()].copy()
    frame = frame.fillna(0)
    frame.index.name = "timestamp"
    frame = frame.reset_index()
    if include_depth and snapshots:
        book_frame = pd.DataFrame.from_dict(snapshots, orient="index").rename_axis("timestamp").reset_index()
        frame = frame.merge(book_frame, on="timestamp", how="left", suffixes=("", "_book"))
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(cache_path, index=False)
    return frame


def _dbn_bars(path: Path, timeframe: str = "1min") -> pd.DataFrame:
    """Stream Databento trade records into bars with contemporaneous TBBO quotes."""
    try:
        import databento as db
    except ImportError as exc:
        raise ValueError("reading .dbn requires the databento package; install requirements.txt") from exc
    store = db.DBNStore.from_file(path)
    if str(store.schema).lower() != "tbbo":
        raise ValueError(f"{path.name} schema is {store.schema}; only TBBO records can be converted to bars here")
    # Databento records expose integer prices in fixed-point units. Convert to
    # `pretty_*` floats and aggregate one bar at a time without DataFrame overhead.
    delta = pd.Timedelta(timeframe)
    if delta <= pd.Timedelta(0):
        raise ValueError(f"invalid DBN bar timeframe: {timeframe}")
    completed = []
    current = None
    records = 0
    last_trade_price = None
    for record in store:
        records += 1
        if str(record.action).upper() != "T":
            continue
        ts = pd.Timestamp(record.ts_event, unit="ns", tz="UTC")
        bucket = ts.floor(delta)
        price = float(record.pretty_price)
        bid, ask = float(record.pretty_bid_px_00), float(record.pretty_ask_px_00)
        if current is not None and bucket != current["timestamp"]:
            completed.append(current)
            current = None
        if current is None:
            current = {"timestamp": bucket, "open": price, "high": price, "low": price, "close": price,
                       "volume": int(record.size), "bid": bid if bid > 0 else np.nan,
                       "ask": ask if ask > 0 else np.nan, "bid_size": int(record.bid_sz_00),
                       "ask_size": int(record.ask_sz_00), "trade_count": 1,
                       "up_ticks": 0, "down_ticks": 0,
                       "flow_buy_volume": 0.0, "flow_sell_volume": 0.0}
        else:
            current["high"] = max(current["high"], price)
            current["low"] = min(current["low"], price)
            current["close"] = price
            current["volume"] += int(record.size)
            current["bid"] = bid if bid > 0 else current["bid"]
            current["ask"] = ask if ask > 0 else current["ask"]
            current["bid_size"] = int(record.bid_sz_00)
            current["ask_size"] = int(record.ask_sz_00)
            current["trade_count"] += 1
        if last_trade_price is not None:
            if price > last_trade_price:
                current["up_ticks"] += 1
            elif price < last_trade_price:
                current["down_ticks"] += 1
        last_trade_price = price
        if bid > 0 and price <= bid:
            current["flow_sell_volume"] += int(record.size)
        elif ask > 0 and price >= ask:
            current["flow_buy_volume"] += int(record.size)
        else:
            aggressor = str(getattr(record, "side", "")).upper()
            if aggressor in {"A", "B"}:
                current["flow_buy_volume" if aggressor == "A" else "flow_sell_volume"] += int(record.size)
        if records % 10_000_000 == 0:
            print(f"DBN decode: {records:,} records scanned", file=sys.stderr, flush=True)
    if current is not None:
        completed.append(current)
    if not completed:
        raise ValueError(f"no trade records found in {path.name}")
    frame = pd.DataFrame(completed)
    frame["flow_delta"] = frame.flow_buy_volume - frame.flow_sell_volume
    frame["flow_delta_pct"] = np.divide(frame.flow_delta, frame.volume,
        out=np.zeros(len(frame), dtype=float), where=frame.volume.to_numpy(dtype=float) != 0) * 100.0
    frame["flow_imbalance"] = np.divide(frame.flow_buy_volume - frame.flow_sell_volume,
        frame.flow_buy_volume + frame.flow_sell_volume,
        out=np.zeros(len(frame), dtype=float),
        where=(frame.flow_buy_volume + frame.flow_sell_volume).to_numpy(dtype=float) != 0)
    return frame


def _imported_tbbo_cache_path(asset: str) -> Path:
    """Canonical cache path shared by TBBO conversion and dataset selection."""
    return ROOT / "data" / "imported_20261030" / "derived" / f"{asset.lower()}_tbbo_1min_v2.parquet"


def _imported_tbbo_bars(asset: str, timeframe: str = "1min") -> pd.DataFrame:
    """Build cached bars and synchronized trade/quote features from imported TBBO parquet.

    Files are processed one session at a time to keep memory use bounded. Identical
    trade rows are retained because they represent separate executions.
    """
    source = ROOT / "data" / "imported_20261030" / "futures_tbbo" / asset
    files = sorted(source.glob("*.parquet"))
    if not files:
        raise ValueError(f"no imported TBBO files for {asset}")
    cache = _imported_tbbo_cache_path(asset)
    latest_source = max(f.stat().st_mtime for f in files)
    if cache.exists() and cache.stat().st_mtime >= latest_source:
        return pd.read_parquet(cache)
    print(f"Converting {asset} TBBO sessions to minute bars ({len(files)} files)...", file=sys.stderr, flush=True)
    columns = ["ts_utc", "price", "size", "aggressor", "bid", "ask", "bid_sz", "ask_sz",
               "is_front", "bad_tick", "no_quote", "crossed", "locked", "off_tick", "outside_quote", "wide_spread"]
    parts: list[pd.DataFrame] = []
    for number, path in enumerate(files, 1):
        frame = pd.read_parquet(path, columns=columns)
        frame = frame.loc[frame["is_front"].fillna(False) & ~frame["bad_tick"].fillna(False)].copy()
        frame = frame.loc[(frame["price"] > 0) & (frame["size"] >= 0)]
        if frame.empty:
            continue
        frame["timestamp"] = pd.to_datetime(frame.pop("ts_utc"), utc=True).dt.floor(timeframe)
        price = pd.to_numeric(frame["price"], errors="coerce")
        last_price = np.nan
        size = pd.to_numeric(frame["size"], errors="coerce").fillna(0)
        aggressor = pd.to_numeric(frame["aggressor"], errors="coerce").fillna(0)
        frame["volume"] = size
        frame["buy_volume"] = size.where(aggressor > 0, 0.0)
        frame["sell_volume"] = size.where(aggressor < 0, 0.0)
        frame["tick_step"] = price.diff()
        if np.isfinite(last_price):
            frame.iloc[0, frame.columns.get_loc("tick_step")] = price.iloc[0] - last_price
        last_price = float(price.iloc[-1])
        frame["up_ticks"] = (frame["tick_step"] > 0).astype("int32")
        frame["down_ticks"] = (frame["tick_step"] < 0).astype("int32")
        frame["trade_count"] = 1
        for flag in ("no_quote", "crossed", "locked", "off_tick", "outside_quote", "wide_spread"):
            frame[f"feed_{flag}_count"] = frame[flag].fillna(False).astype("int32")
        frame["spread"] = pd.to_numeric(frame["ask"], errors="coerce") - pd.to_numeric(frame["bid"], errors="coerce")
        frame["mbo_microprice"] = np.divide(
            pd.to_numeric(frame["ask"], errors="coerce") * pd.to_numeric(frame["bid_sz"], errors="coerce") +
            pd.to_numeric(frame["bid"], errors="coerce") * pd.to_numeric(frame["ask_sz"], errors="coerce"),
            pd.to_numeric(frame["bid_sz"], errors="coerce") + pd.to_numeric(frame["ask_sz"], errors="coerce"),
            out=np.full(len(frame), np.nan),
            where=(pd.to_numeric(frame["bid_sz"], errors="coerce") + pd.to_numeric(frame["ask_sz"], errors="coerce")).to_numpy() > 0,
        )
        grouped = frame.groupby("timestamp", sort=True)
        agg = grouped.agg(
            open=("price", "first"), high=("price", "max"), low=("price", "min"), close=("price", "last"),
            volume=("volume", "sum"), trade_count=("trade_count", "sum"), bid=("bid", "last"), ask=("ask", "last"),
            bid_size=("bid_sz", "last"), ask_size=("ask_sz", "last"), up_ticks=("up_ticks", "sum"), down_ticks=("down_ticks", "sum"),
            flow_buy_volume=("buy_volume", "sum"), flow_sell_volume=("sell_volume", "sum"),
            feed_spread=("spread", "last"), feed_microprice=("mbo_microprice", "last"),
            feed_no_quote_count=("feed_no_quote_count", "sum"), feed_crossed_count=("feed_crossed_count", "sum"),
            feed_locked_count=("feed_locked_count", "sum"), feed_off_tick_count=("feed_off_tick_count", "sum"),
            feed_outside_quote_count=("feed_outside_quote_count", "sum"), feed_wide_spread_count=("feed_wide_spread_count", "sum"),
        ).reset_index()
        agg["flow_delta"] = agg.flow_buy_volume - agg.flow_sell_volume
        agg["flow_delta_pct"] = np.divide(agg.flow_delta, agg.volume, out=np.zeros(len(agg)), where=agg.volume.to_numpy() != 0) * 100
        flow_total = agg.flow_buy_volume + agg.flow_sell_volume
        agg["flow_imbalance"] = np.divide(agg.flow_delta, flow_total, out=np.zeros(len(agg)), where=flow_total.to_numpy() != 0)
        parts.append(agg)
        progress_stride = max(1, math.ceil(len(files) / 20))
        if number % progress_stride == 0 or number == len(files):
            print(f"{asset} TBBO conversion: {number}/{len(files)} sessions", file=sys.stderr, flush=True)
    if not parts:
        raise ValueError(f"no valid imported TBBO trades for {asset}")
    combined = pd.concat(parts, ignore_index=True).sort_values("timestamp", kind="stable")
    aggregation = {
        "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum",
        "trade_count": "sum", "bid": "last", "ask": "last", "bid_size": "last", "ask_size": "last",
        "up_ticks": "sum", "down_ticks": "sum", "flow_buy_volume": "sum", "flow_sell_volume": "sum",
        "feed_spread": "last", "feed_microprice": "last", "feed_no_quote_count": "sum", "feed_crossed_count": "sum",
        "feed_locked_count": "sum", "feed_off_tick_count": "sum", "feed_outside_quote_count": "sum",
        "feed_wide_spread_count": "sum",
    }
    result = combined.groupby("timestamp", as_index=False, sort=True).agg(aggregation)
    result["flow_delta"] = result.flow_buy_volume - result.flow_sell_volume
    result["flow_delta_pct"] = np.divide(result.flow_delta, result.volume, out=np.zeros(len(result)),
                                          where=result.volume.to_numpy() != 0) * 100.0
    flow_total = result.flow_buy_volume + result.flow_sell_volume
    result["flow_imbalance"] = np.divide(result.flow_delta, flow_total, out=np.zeros(len(result)),
                                          where=flow_total.to_numpy() != 0)
    cache.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(cache, index=False)
    return result


def _attach_nq_mbp10(frame: pd.DataFrame) -> pd.DataFrame:
    """Join 5-second NQ MBP-10 summaries to minute bars without using future snapshots."""
    path = ROOT / "data" / "imported_20261030" / "l2_book" / "NQ_MBP10" / "NQ_MBP10_5s_clean.parquet"
    if not path.exists() or frame.empty:
        return frame
    book = pd.read_parquet(path)
    if isinstance(book.index, pd.DatetimeIndex) and book.index.name and book.index.name not in book.columns:
        book = book.reset_index()
    timestamp_column = next((name for name in ("ts", "timestamp", "ts_utc", "ts_ny") if name in book.columns), None)
    if timestamp_column is None:
        print("NQ MBP-10 summary has no within-session timestamps; skipping unsynchronized book features.",
              file=sys.stderr, flush=True)
        return frame
    ts = pd.to_datetime(book[timestamp_column], errors="coerce")
    if ts.dt.tz is None:
        ts = ts.dt.tz_localize("America/New_York", ambiguous="infer", nonexistent="shift_forward")
    book["timestamp"] = ts.dt.tz_convert("UTC").dt.floor("1min")
    rename = {
        "mid_open": "mbo_mid_open", "mid_high": "mbo_mid_high", "mid_low": "mbo_mid_low", "mid_close": "mbo_mid_close",
        "mid_change": "mbo_mid_change", "updates": "mbo_updates", "add_count": "mbo_add_count",
        "cancel_count": "mbo_cancel_count", "modify_count": "mbo_modify_count", "bid_side_count": "mbo_bid_side_count",
        "ask_side_count": "mbo_ask_side_count", "neutral_side_count": "mbo_neutral_side_count",
        "avg_spread": "mbo_avg_spread", "last_spread": "mbo_mbp10_spread", "avg_microprice": "mbo_avg_microprice",
        "last_microprice": "mbo_mbp10_microprice", "avg_top_imbalance": "mbo_avg_top_imbalance", "last_top_imbalance": "mbo_top_imbalance",
        "avg_depth_5_bid": "mbo_avg_depth_5_bid", "last_depth_5_bid": "mbo_bid_depth_5",
        "avg_depth_5_ask": "mbo_avg_depth_5_ask", "last_depth_5_ask": "mbo_ask_depth_5",
        "avg_depth_5_imbalance": "mbo_avg_depth_imbalance_5", "last_depth_5_imbalance": "mbo_depth_imbalance_5",
        "avg_depth_10_bid": "mbo_avg_depth_10_bid", "last_depth_10_bid": "mbo_bid_depth_10",
        "avg_depth_10_ask": "mbo_avg_depth_10_ask", "last_depth_10_ask": "mbo_ask_depth_10",
        "avg_depth_10_imbalance": "mbo_avg_depth_imbalance_10", "last_depth_10_imbalance": "mbo_depth_imbalance_10",
        "avg_book_pressure": "mbo_avg_book_pressure", "last_book_pressure": "mbo_book_pressure",
    }
    book = book.rename(columns=rename)
    sum_names = [c for c in book.columns if c in {"mbo_updates", "mbo_add_count", "mbo_cancel_count", "mbo_modify_count",
                  "mbo_bid_side_count", "mbo_ask_side_count", "mbo_neutral_side_count"}]
    last_names = [c for c in book.columns if c.startswith("mbo_") and c not in sum_names and c not in {"mbo_mid_open", "mbo_mid_high", "mbo_mid_low"}]
    agg = {c: "sum" for c in sum_names}
    agg.update({c: "last" for c in last_names})
    for col in ("mbo_mid_open", "mbo_mid_high", "mbo_mid_low"):
        agg[col] = {"mbo_mid_open": "first", "mbo_mid_high": "max", "mbo_mid_low": "min"}[col]
    book = book.groupby("timestamp", as_index=False, sort=True).agg(agg)
    return frame.merge(book, on="timestamp", how="left", validate="one_to_one")


def _imported_mgc_bbo_bars() -> pd.DataFrame:
    """Create minute midpoint proxy bars and top-of-book features from the MGC quote feed."""
    path = ROOT / "data" / "imported_20261030" / "l2_book" / "MGC" / "MGC_BBO.parquet"
    cache = path.parent / "mgc_bbo_midpoint_1min_v3.parquet"
    event_files = sorted(path.parent.glob("MGC_L2_events_*.parquet"))
    latest_source = max([path.stat().st_mtime, *(f.stat().st_mtime for f in event_files)])
    if cache.exists() and cache.stat().st_mtime >= latest_source:
        return pd.read_parquet(cache)
    book = pd.read_parquet(path)
    if isinstance(book.index, pd.DatetimeIndex) and book.index.name and book.index.name not in book.columns:
        book = book.reset_index()
    book["timestamp"] = pd.to_datetime(book.pop("ts"), utc=True).dt.floor("1min")
    book = book.loc[book.valid.fillna(False) & ~book.halt.fillna(False) & ~book.crossed_or_locked.fillna(False)].copy()
    book["mid"] = (book.bid_px + book.ask_px) / 2
    book["mbo_spread"] = book.ask_px - book.bid_px
    total = book.bid_sz + book.ask_sz
    book["mbo_top_imbalance"] = np.divide(book.bid_sz - book.ask_sz, total, out=np.zeros(len(book)), where=total.to_numpy() != 0)
    depth = book.bid_sz5 + book.ask_sz5
    book["mbo_depth_imbalance_5"] = np.divide(book.bid_sz5 - book.ask_sz5, depth, out=np.zeros(len(book)), where=depth.to_numpy() != 0)
    book["mbo_microprice"] = np.divide(book.ask_px * book.bid_sz + book.bid_px * book.ask_sz, total,
                                       out=np.full(len(book), np.nan), where=total.to_numpy() != 0)
    book["open"] = book["mid"]
    book["high"] = book["mid"]
    book["low"] = book["mid"]
    book["close"] = book["mid"]
    result = book.groupby("timestamp", as_index=False, sort=True).agg(
        open=("open", "first"), high=("high", "max"), low=("low", "min"), close=("close", "last"),
        volume=("bid_sz5", "size"), bid=("bid_px", "last"), ask=("ask_px", "last"),
        bid_size=("bid_sz", "last"), ask_size=("ask_sz", "last"), mbo_bid_depth_5=("bid_sz5", "last"),
        mbo_ask_depth_5=("ask_sz5", "last"), mbo_spread=("mbo_spread", "last"),
        mbo_microprice=("mbo_microprice", "last"), mbo_top_imbalance=("mbo_top_imbalance", "last"),
        mbo_depth_imbalance_5=("mbo_depth_imbalance_5", "last"), crossed_or_locked=("crossed_or_locked", "max"),
    )
    event_parts = []
    for event_number, event_path in enumerate(event_files, 1):
        print(f"MGC book conversion: event file {event_number}/{len(event_files)} ({event_path.name})",
              file=sys.stderr, flush=True)
        events = pd.read_parquet(event_path, columns=["ts", "side", "op", "level"])
        events["timestamp"] = pd.to_datetime(events.pop("ts"), utc=True).dt.floor("1min")
        events["side"] = events["side"].astype("string").str.lower()
        events["op"] = events["op"].astype("string").str.lower()
        events["level"] = pd.to_numeric(events["level"], errors="coerce").fillna(-1).astype("int16")
        by_op_side = events.groupby(["timestamp", "op", "side"], observed=True).size().unstack(["op", "side"], fill_value=0)
        by_op_side.columns = [f"mbo_{op}_{side}_count" for op, side in by_op_side.columns]
        for op in ("add", "cancel", "modify"):
            for side in ("bid", "ask"):
                name = f"mbo_{op}_{side}_count"
                if name not in by_op_side:
                    by_op_side[name] = 0
        events = events.loc[events.level.between(0, 10)]
        by_level = events.groupby(["timestamp", "level"], observed=True).size().unstack("level", fill_value=0)
        by_level = by_level.rename(columns={level: f"mbo_level_{level}_event_count" for level in by_level.columns})
        for level in range(11):
            name = f"mbo_level_{level}_event_count"
            if name not in by_level:
                by_level[name] = 0
        event_parts.append(by_op_side.join(by_level, how="outer").fillna(0).reset_index())
        del events
    if event_parts:
        events_by_minute = pd.concat(event_parts, ignore_index=True).groupby("timestamp", as_index=False, sort=True).sum(numeric_only=True)
        result = result.merge(events_by_minute, on="timestamp", how="left", validate="one_to_one")
        for field in events_by_minute.columns:
            if field != "timestamp":
                result[field] = result[field].fillna(0)
    result["volume"] = 0.0  # no reliable executed-trade volume in this quote dataset
    result["data_is_quote_midpoint_proxy"] = True
    cache.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(cache, index=False)
    return result


def _attach_mgc_book(frame: pd.DataFrame) -> pd.DataFrame:
    """Attach available MGC L2 features to trade-derived bars (5-day overlap only)."""
    book = _imported_mgc_bbo_bars()
    features = ["timestamp"] + [name for name in book.columns if name.startswith("mbo_")]
    return frame.merge(book[features], on="timestamp", how="left", validate="one_to_one")


def load_data(path: Path, naive_timezone: str = "UTC", include_depth: bool = False) -> pd.DataFrame:
    if path.is_dir():
        frame = _mbo_event_bars(path, include_depth=include_depth)
    elif path.suffix.lower() in {".csv", ".gz"}:
        header = pd.read_csv(path, nrows=0, compression="gzip" if path.suffix.lower() == ".gz" else None)
        if {"timestamp", "action", "side", "price", "size"}.issubset({str(c).strip().lower() for c in header.columns}):
            frame = _mbo_event_bars(path, include_depth=include_depth)
        elif path.suffix.lower() == ".gz":
            frame = pd.read_csv(path, compression="gzip", low_memory=False)
        else:
            frame = pd.read_csv(path, low_memory=False)
    suffix = path.suffix.lower()
    if path.is_dir():
        pass
    elif suffix == ".dbn":
        cache_path = path.with_suffix(".1min.parquet")
        if cache_path.exists() and cache_path.stat().st_mtime >= path.stat().st_mtime:
            cached = pd.read_parquet(cache_path)
            if {"flow_buy_volume", "flow_sell_volume", "flow_delta", "up_ticks", "down_ticks"}.issubset(cached.columns):
                frame = cached
            else:
                frame = _dbn_bars(path)
                frame.to_parquet(cache_path, index=False)
        else:
            print(f"Converting DBN stream to cached bars: {cache_path.name}", file=sys.stderr, flush=True)
            frame = _dbn_bars(path)
            frame.to_parquet(cache_path, index=False)
    elif suffix == ".parquet":
        frame = pd.read_parquet(path)
    elif suffix in {".csv", ".gz"}:
        pass
    else:
        frame = pd.read_csv(path, low_memory=False)
    if "timestamp" not in frame.columns and "datetime" not in frame.columns and isinstance(frame.index, pd.DatetimeIndex):
        frame = frame.reset_index()
    normalized = [str(c).strip().lower() for c in frame.columns]
    duplicates = sorted({c for c in normalized if normalized.count(c) > 1})
    if duplicates:
        raise ValueError(f"duplicate column names in {path.name}: {', '.join(duplicates)}")
    frame.columns = normalized
    # Many futures exports use YYYYMMDD and HHMMSS as separate integer columns.
    # Combine them before applying aliases; renaming both to `timestamp` caused
    # pandas' confusing "cannot assemble with duplicate keys" exception.
    if "timestamp" not in frame.columns and "datetime" not in frame.columns and "date" in frame.columns:
        if "time" in frame.columns:
            dates = frame["date"].astype("string").str.strip().str.replace(r"\.0$", "", regex=True)
            times = frame["time"].astype("string").str.strip().str.replace(r"\.0$", "", regex=True)
            combined = dates + " " + times
            # OFB export uses MM/DD/YYYY + HH:MM; do not digit-strip HH:MM,
            # which previously interpreted 18:01 as 00:18:01.
            parsed = pd.to_datetime(combined, format="%m/%d/%Y %H:%M:%S", errors="coerce")
            parsed = parsed.fillna(pd.to_datetime(combined, format="%m/%d/%Y %H:%M", errors="coerce"))
            compact_times = times.str.replace(r"[^0-9]", "", regex=True).str.zfill(6)
            compact = dates + " " + compact_times
            fallback = pd.to_datetime(compact, format="%Y%m%d %H%M%S", errors="coerce")
            parsed = parsed.fillna(fallback).fillna(pd.to_datetime(combined, errors="coerce"))
            frame["timestamp"] = parsed
            frame = frame.drop(columns=["date", "time"])
        else:
            frame = frame.rename(columns={"date": "timestamp"})
    aliases = {"datetime": "timestamp", "vol": "volume", "bid_px_00": "bid", "ask_px_00": "ask",
               "bid_sz_00": "bid_size", "ask_sz_00": "ask_size", "ts_event": "timestamp", "price": "close",
               "upvol": "up_volume", "downvol": "down_volume"}
    frame = frame.rename(columns=aliases)
    if frame.columns.duplicated().any():
        dup = sorted(set(frame.columns[frame.columns.duplicated(keep=False)]))
        raise ValueError(f"columns map to the same field in {path.name}: {', '.join(dup)}")
    required = {"timestamp", "open", "high", "low", "close"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"missing data columns: {', '.join(sorted(missing))}")
    if "volume" not in frame:
        frame["volume"] = 0.0
    parsed_timestamp = pd.to_datetime(frame.timestamp, errors="coerce")
    if isinstance(parsed_timestamp.dtype, pd.DatetimeTZDtype):
        frame["timestamp"] = parsed_timestamp.dt.tz_convert("UTC")
    elif pd.api.types.is_datetime64_dtype(parsed_timestamp.dtype):
        try:
            localized = parsed_timestamp.dt.tz_localize(naive_timezone, ambiguous="infer", nonexistent="shift_forward")
        except (ValueError, TypeError):
            localized = parsed_timestamp.dt.tz_localize(naive_timezone, ambiguous=True, nonexistent="shift_forward")
        frame["timestamp"] = localized.dt.tz_convert("UTC")
    else:
        frame["timestamp"] = pd.to_datetime(frame.timestamp, utc=True, errors="coerce")
    for col in ("open", "high", "low", "close", "volume"):
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    valid = frame[["timestamp", "open", "high", "low", "close"]].notna().all(axis=1)
    if not bool(valid.all()):
        frame = frame.loc[valid].copy()
    if not frame["timestamp"].is_monotonic_increasing:
        frame = frame.sort_values("timestamp", kind="stable")
    # Avoid an unconditional full-frame hash/deduplicate copy for large clean files.
    duplicate_timestamps = frame["timestamp"].duplicated(keep="last")
    if bool(duplicate_timestamps.any()):
        frame = frame.loc[~duplicate_timestamps]
    frame = frame.reset_index(drop=True)
    if frame.empty:
        raise ValueError("no valid OHLC rows found")
    if (frame.high < frame.low).any():
        raise ValueError("found bars where high is below low")
    if {"up_volume", "down_volume"}.issubset(frame.columns):
        frame["up_volume"] = pd.to_numeric(frame.up_volume, errors="coerce").fillna(0.0)
        frame["down_volume"] = pd.to_numeric(frame.down_volume, errors="coerce").fillna(0.0)
        frame["flow_buy_volume"] = frame.up_volume
        frame["flow_sell_volume"] = frame.down_volume
        frame["flow_delta"] = frame.up_volume - frame.down_volume
        frame["flow_delta_pct"] = np.divide(frame.flow_delta, frame.volume,
            out=np.zeros(len(frame), dtype=float), where=frame.volume.to_numpy(dtype=float) != 0) * 100.0
        flow_total = frame.up_volume + frame.down_volume
        frame["flow_imbalance"] = np.divide(frame.flow_delta, flow_total,
            out=np.zeros(len(frame), dtype=float), where=flow_total.to_numpy(dtype=float) != 0)
    return frame


def resample_primary(data: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Aggregate input bars to the strategy's primary chart interval."""
    source_period = data.timestamp.diff().median()
    target_period = pd.Timedelta(timeframe)
    if pd.isna(source_period) or target_period <= source_period:
        return data
    aggregation = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    for field in ("bid", "ask", "bid_size", "ask_size", "trade_count"):
        if field in data.columns:
            aggregation[field] = "last" if field in {"bid", "ask", "bid_size", "ask_size"} else "sum"
    for field in ("up_ticks", "down_ticks"):
        if field in data.columns:
            aggregation[field] = "sum"
    for field in data.columns:
        if str(field).startswith("mbo_"):
            if field == "mbo_mid_open":
                aggregation[field] = "first"
            elif field == "mbo_mid_high":
                aggregation[field] = "max"
            elif field == "mbo_mid_low":
                aggregation[field] = "min"
            elif field == "mbo_mid_change":
                aggregation[field] = "sum"
            elif field.startswith("mbo_avg_"):
                aggregation[field] = "mean"
            elif re.search(r"(?:_count|_events|_updates)$", str(field)) or field in {"mbo_updates"}:
                aggregation[field] = "sum"
            else:
                aggregation[field] = "last"
        elif str(field).startswith("flow_"):
            aggregation[field] = "sum" if any(str(field).endswith(suffix) for suffix in ("_volume", "_count", "_delta")) else "last"
        elif str(field).startswith("liq_"):
            aggregation[field] = "last"
        elif str(field).startswith("feed_"):
            aggregation[field] = "sum"
        elif str(field).startswith("option_"):
            aggregation[field] = "sum" if any(str(field).endswith(suffix) for suffix in ("_volume", "_count", "_premium")) else "last"
    frame = (data.set_index("timestamp")
             .resample(timeframe, closed="left", label="left")
             .agg(aggregation)
             .dropna(subset=["open", "high", "low", "close"]))
    if {"flow_buy_volume", "flow_sell_volume"}.issubset(frame.columns):
        frame["flow_delta"] = frame.flow_buy_volume - frame.flow_sell_volume
        frame["flow_delta_pct"] = np.divide(frame.flow_delta, frame.volume,
            out=np.zeros(len(frame), dtype=float), where=frame.volume.to_numpy(dtype=float) != 0) * 100.0
        frame["flow_imbalance"] = np.divide(frame.flow_buy_volume - frame.flow_sell_volume,
            frame.flow_buy_volume + frame.flow_sell_volume,
            out=np.zeros(len(frame), dtype=float),
            where=(frame.flow_buy_volume + frame.flow_sell_volume).to_numpy(dtype=float) != 0)
    frame.index.name = "timestamp"
    return frame.reset_index()


def filter_session(data: pd.DataFrame, session: str, timezone: str) -> pd.DataFrame:
    """Restrict Data1 to a clock-time session such as 0930-1600."""
    match = re.fullmatch(r"\s*(\d{3,4})\s*-\s*(\d{3,4})\s*", session)
    if not match:
        raise ValueError("Data1 session must use HHMM-HHMM, for example 0930-1600")
    start, end = (int(value) for value in match.groups())
    start_min = (start // 100) * 60 + start % 100
    end_min = (end // 100) * 60 + end % 100
    local = pd.DatetimeIndex(data.timestamp).tz_convert(timezone)
    minute = local.hour * 60 + local.minute
    if start_min < end_min:
        mask = (minute >= start_min) & (minute < end_min)
    else:
        mask = (minute >= start_min) | (minute < end_min)
    result = data.loc[mask].reset_index(drop=True)
    if result.empty:
        raise ValueError(f"no source bars fall within Data1 session {session} in {timezone}")
    return result


def infer_strategy_settings(source: str) -> dict:
    """Read common chart requirements from PowerLanguage header comments."""
    settings = {}
    match = re.search(r"(?im)^\s*//\s*ENGINE_BAR_TIMEFRAME\s*=\s*(\S+)\s*$", source)
    if match:
        settings["bar_timeframe"] = match.group(1)
    else:
        match = re.search(r"(?i)Recommended\s+chart\s*:\s*(\d+)\s*[- ]?\s*(min|minute)\s+bars", source)
        if match:
            settings["bar_timeframe"] = f"{int(match.group(1))}min"
        else:
            match = re.search(r"(?im)^\s*Data1\s*=\s*[^\n]*?\b(\d+)\s*[- ]?\s*(hour|hr|minute|min)\s+chart", source)
            if match:
                unit = "h" if match.group(2).lower() in {"hour", "hr"} else "min"
                settings["bar_timeframe"] = f"{int(match.group(1))}{unit}"
    match = re.search(r"(?im)^\s*//\s*ENGINE_DATA2_TIMEFRAME\s*=\s*(\S+)\s*$", source)
    if match:
        settings["data2_timeframe"] = match.group(1)
    else:
        match = re.search(r"(?im)^\s*//\s*Data2\s*:\s*.*?\b(daily|day)\b", source)
        if not match:
            match = re.search(r"(?im)^\s*Data2\s*=\s*[^\n]*?\b(daily|day)\s+chart", source)
        if match:
            settings["data2_timeframe"] = "1D"
    match = re.search(r"(?im)^\s*//\s*ENGINE_SESSION_TIMEZONE\s*=\s*(\S+)\s*$", source)
    if match:
        settings["session_timezone"] = match.group(1)
    elif re.search(r"(?i)time\s+zone\s*:\s*EXCHANGE\s*\(\s*US\s+Central\s*\)|all\s+time\s+inputs\s+below\s+are\s+Central\s+time", source):
        settings["session_timezone"] = "America/Chicago"
    elif re.search(r"(?i)\bET\b|Eastern\s+Time|New\s+York", source):
        settings["session_timezone"] = "America/New_York"
    match = re.search(r"(?im)^\s*//\s*ENGINE_DATA_TIMEZONE\s*=\s*(\S+)\s*$", source)
    if match:
        settings["data_timezone"] = match.group(1)
    match = re.search(r"(?im)^\s*//\s*ENGINE_DATA1_SESSION\s*=\s*(\d{3,4}\s*-\s*\d{3,4})\s*$", source)
    if match:
        settings["data1_session"] = re.sub(r"\s+", "", match.group(1))
    else:
        match = re.search(r"(?i)RTH\s+session\s+(\d{3,4})\s*[-–]\s*(\d{3,4})\s*(ET|Eastern\s+Time)", source)
        if match:
            settings["data1_session"] = f"{int(match.group(1)):04d}-{int(match.group(2)):04d}"
    return settings


def calc_stats(trades, equity, initial_cash, session_timezone="UTC"):
    e = equity.set_index("timestamp")["equity"].astype(float)
    index = pd.DatetimeIndex(e.index)
    if index.tz is not None:
        daily_equity = e.copy()
        daily_equity.index = index.tz_convert(session_timezone)
    else:
        daily_equity = e.copy()
    daily_equity = daily_equity.resample("1D").last().dropna()
    if len(daily_equity):
        close_returns = daily_equity.pct_change().dropna().to_numpy(dtype=float)
        first_return = float(daily_equity.iloc[0] / initial_cash - 1.0)
        rets = np.r_[first_return, close_returns]
        rets = rets[np.isfinite(rets)]
    else:
        rets = np.array([], dtype=float)
    pnl = trades.pnl.astype(float) if len(trades) else pd.Series(dtype=float)
    winners, losers = pnl[pnl > 0], pnl[pnl < 0]
    years = max((e.index[-1] - e.index[0]).total_seconds() / (365.25 * 86400), 1 / 365.25)
    peak = e.cummax()
    dd = e - peak
    dd_pct = e / peak - 1
    # Risk ratios use the same close-to-close daily returns as the downside deviation.
    periods_per_year = max(len(rets) / years, 1.0)
    daily_std = float(np.std(rets, ddof=1)) if len(rets) > 1 else 0.0
    vol = daily_std * math.sqrt(periods_per_year)
    sharpe = float(np.mean(rets) / daily_std * math.sqrt(periods_per_year)) if daily_std > 0 else 0.0
    downside_deviation = float(np.sqrt(np.mean(np.minimum(rets, 0.0) ** 2))) if len(rets) else 0.0
    sortino = float(np.mean(rets) / downside_deviation * math.sqrt(periods_per_year)) if downside_deviation > 0 else None
    duration = (dd < 0).groupby((dd == 0).cumsum()).sum().max() if len(dd) else 0
    total_return = float(e.iloc[-1] / initial_cash - 1)
    cagr = float((e.iloc[-1] / initial_cash) ** (1 / years) - 1) if e.iloc[-1] > 0 else -1.0
    gross_profit, gross_loss = float(winners.sum()), float(-losers.sum())
    win_streak = loss_streak = max_win_streak = max_loss_streak = 0
    for result in pnl:
        if result > 0:
            win_streak += 1
            loss_streak = 0
        elif result < 0:
            loss_streak += 1
            win_streak = 0
        else:
            win_streak = loss_streak = 0
        max_win_streak = max(max_win_streak, win_streak)
        max_loss_streak = max(max_loss_streak, loss_streak)
    price_returns = trades["price_return_pct"].astype(float) if len(trades) and "price_return_pct" in trades else pd.Series(dtype=float)
    compound_price_return = float((np.prod(1.0 + price_returns.to_numpy() / 100.0) - 1.0) * 100.0) if len(price_returns) else 0.0
    return {
        "initial_cash": initial_cash, "final_equity": float(e.iloc[-1]), "net_profit": float(e.iloc[-1] - initial_cash),
        "total_return_pct": total_return * 100, "cagr_pct": cagr * 100, "annualized_volatility_pct": vol * 100,
        "sharpe_ratio": sharpe, "sortino_ratio": sortino,
        "risk_ratio_return_basis": "daily close-to-close equity returns; Sortino downside deviation uses all daily periods with a 0% minimum acceptable return",
        "annualization_periods_per_year": periods_per_year,
        "max_drawdown": float(dd.min()), "max_drawdown_pct": float(dd_pct.min() * 100),
        "max_drawdown_duration_bars": int(duration), "calmar_ratio": cagr / abs(float(dd_pct.min())) if dd_pct.min() < 0 else None,
        "trade_count": int(len(pnl)), "win_rate_pct": float((pnl > 0).mean() * 100) if len(pnl) else 0,
        "profit_factor": gross_profit / gross_loss if gross_loss else (None if not gross_profit else "infinite"),
        "average_winner": float(winners.mean()) if len(winners) else 0,
        "average_loser": float(losers.mean()) if len(losers) else 0,
        "payoff_ratio": float(winners.mean() / abs(losers.mean())) if len(winners) and len(losers) and losers.mean() else None,
        "recovery_factor": float((e.iloc[-1] - initial_cash) / abs(float(dd.min()))) if dd.min() < 0 else None,
        "max_consecutive_wins": max_win_streak, "max_consecutive_losses": max_loss_streak,
        "expectancy_per_trade": float(pnl.mean()) if len(pnl) else 0,
        "average_trade": float(pnl.mean()) if len(pnl) else 0, "median_trade": float(pnl.median()) if len(pnl) else 0,
        "gross_profit": gross_profit, "gross_loss": gross_loss,
        "best_trade": float(pnl.max()) if len(pnl) else 0, "worst_trade": float(pnl.min()) if len(pnl) else 0,
        "average_holding_bars": float(trades.bars.mean()) if len(trades) else 0,
        "mean_trade_price_return_pct": float(price_returns.mean()) if len(price_returns) else 0.0,
        "median_trade_price_return_pct": float(price_returns.median()) if len(price_returns) else 0.0,
        "compounded_trade_price_return_pct": compound_price_return,
        "exit_reason_counts": {str(k): int(v) for k, v in trades.exit_reason.value_counts().items()} if len(trades) else {},
        "market_exposure_pct": float((equity.position != 0).mean() * 100),
        "bars": int(len(e)), "start": str(e.index[0]), "end": str(e.index[-1]),
    }

def monte_carlo(equity, initial_cash, runs, seed, block_days=5, sample_paths=20):
    """Block-bootstrap daily equity returns into complete simulated equity paths."""
    if runs < 1:
        raise ValueError("Monte Carlo runs must be at least 1")
    if block_days < 1:
        raise ValueError("Monte Carlo block length must be at least 1 day")

    daily_equity = (equity.set_index("timestamp")["equity"].astype(float)
                    .resample("1D").last().dropna())
    observed = daily_equity.to_numpy(dtype=np.float64)
    if not len(observed):
        observed = np.array([float(initial_cash)])

    # Include the first observed daily change relative to starting capital, then
    # use daily close-to-close returns for the remaining observations.
    previous = np.r_[float(initial_cash), observed[:-1]]
    returns = np.divide(observed - previous, previous, out=np.zeros_like(observed),
                        where=np.abs(previous) > 1e-12)
    returns = np.nan_to_num(returns, nan=0.0, posinf=0.0, neginf=-0.999999)
    returns = np.maximum(returns, -0.999999)
    horizon = len(returns)
    block_days = min(int(block_days), horizon)
    block_count = int(math.ceil(horizon / block_days))
    rng = np.random.default_rng(seed)

    paths = np.empty((runs, horizon + 1), dtype=np.float64)
    paths[:, 0] = float(initial_cash)
    batch_size = max(1, min(10, runs // 20 or 1))
    progress_step = max(1, runs // 20)
    offsets = np.arange(block_days, dtype=np.int64)
    for start in range(0, runs, batch_size):
        stop = min(start + batch_size, runs)
        batch = stop - start
        starts = rng.integers(0, horizon, size=(batch, block_count))
        # Circular moving blocks preserve short-range return clustering while
        # allowing every historical date to be sampled as a block start.
        indices = (starts[:, :, None] + offsets[None, None, :]) % horizon
        sampled = returns[indices.reshape(batch, -1)[:, :horizon]]
        curves = float(initial_cash) * np.cumprod(1.0 + sampled, axis=1)
        paths[start:stop, 1:] = curves
        if stop == runs or stop // progress_step > start // progress_step:
            print(f"Monte Carlo progress: {stop:,} simulated equity curves "
                  f"({stop * 100 / runs:.0f}%)", file=sys.stderr, flush=True)

    peaks = np.maximum.accumulate(paths, axis=1)
    drawdowns = np.max(peaks - paths, axis=1)
    terminals = paths[:, -1]
    pct = np.percentile(paths, [5, 25, 50, 75, 95], axis=0)
    bands = pd.DataFrame({
        "simulation_day": np.arange(horizon + 1),
        "actual_equity": np.r_[float(initial_cash), observed],
        "equity_p05": pct[0], "equity_p25": pct[1], "equity_median": pct[2],
        "equity_p75": pct[3], "equity_p95": pct[4],
    })
    keep = min(max(0, int(sample_paths)), runs)
    samples = pd.DataFrame({"simulation_day": np.arange(horizon + 1)})
    for idx in range(keep):
        samples[f"simulation_{idx + 1:02d}"] = paths[idx]

    summary = {
        "method": "daily percentage-return circular moving-block bootstrap",
        "runs": int(runs), "horizon_days": int(horizon), "block_length_days": int(block_days),
        "terminal_equity_p05": float(np.percentile(terminals, 5)),
        "terminal_equity_median": float(np.median(terminals)),
        "terminal_equity_p95": float(np.percentile(terminals, 95)),
        "probability_of_loss_pct": float((terminals < initial_cash).mean() * 100),
        "probability_of_ruin_pct": float((paths.min(axis=1) <= 0).mean() * 100),
        "max_drawdown_currency_p50": float(np.percentile(drawdowns, 50)),
        "max_drawdown_currency_p95": float(np.percentile(drawdowns, 95)),
    }
    return summary, terminals, drawdowns, bands, samples, paths


def parse_data_period(value: str) -> pd.DateOffset | None:
    text = value.strip().lower()
    if text in {"all", "full", "everything"}:
        return None
    match = re.fullmatch(r"(\d+)\s*(y|yr|yrs|year|years|mo|mos|month|months|w|wk|wks|week|weeks|d|day|days)", text)
    if not match:
        raise ValueError("data period must be 'all' or a duration such as 4y, 18mo, 12w, or 90d")
    amount = int(match.group(1))
    unit = match.group(2)
    if amount < 1:
        raise ValueError("data period must be greater than zero")
    if unit in {"y", "yr", "yrs", "year", "years"}:
        return pd.DateOffset(years=amount)
    if unit in {"mo", "mos", "month", "months"}:
        return pd.DateOffset(months=amount)
    if unit in {"w", "wk", "wks", "week", "weeks"}:
        return pd.DateOffset(weeks=amount)
    return pd.DateOffset(days=amount)


def limit_data_period(data: pd.DataFrame, value: str) -> pd.DataFrame:
    offset = parse_data_period(value)
    if offset is None:
        return data
    cutoff = data.timestamp.iloc[-1] - offset
    selected = data.loc[data.timestamp >= cutoff].reset_index(drop=True)
    if selected.empty:
        raise ValueError(f"no data falls inside requested period {value!r}")
    return selected


def make_lucidflex_pnl_paths(trades, data, point_value, commission, account_size, timezone,
                             timeframe, runs, seed, block_days=5,
                             commission_rate_per_side=0.0, pnl_basis="points"):
    """Bootstrap trade P&L with an explicit point-move or current-notional basis."""
    if runs < 1 or block_days < 1:
        raise ValueError("LucidFlex Monte Carlo runs and block length must be positive")
    latest_price = float(data.close.iloc[-1])
    if len(trades):
        prop_trades = trades.copy()
        entry_times = pd.to_datetime(prop_trades.entry_time, utc=True).dt.tz_convert(timezone)
        exit_times = pd.to_datetime(prop_trades.exit_time, utc=True).dt.tz_convert(timezone)
        bar_period = pd.Timedelta(timeframe)
        bar_starts = pd.DatetimeIndex(pd.to_datetime(data.timestamp, utc=True))
        bar_ends = bar_starts + bar_period
        cutoff_delta = pd.Timedelta(hours=16, minutes=45)
        for row_idx, (entry_time, exit_time) in enumerate(zip(entry_times, exit_times)):
            cutoff = entry_time.normalize() + cutoff_delta
            if entry_time >= cutoff:
                cutoff = cutoff + pd.DateOffset(days=1)
            while cutoff.weekday() >= 5:
                cutoff = cutoff + pd.DateOffset(days=1)
            if exit_time + bar_period > cutoff:
                cutoff_utc = cutoff.tz_convert("UTC")
                open_idx = int(bar_starts.searchsorted(cutoff_utc, side="left"))
                if open_idx < len(data) and bar_starts[open_idx] == cutoff_utc:
                    flat_price = float(data.open.iloc[open_idx])
                else:
                    close_idx = int(bar_ends.searchsorted(cutoff_utc, side="right") - 1)
                    if close_idx < 0 or bar_ends[close_idx] <= entry_times.iloc[row_idx].tz_convert("UTC"):
                        continue
                    flat_price = float(data.close.iloc[close_idx])
                prop_trades.loc[prop_trades.index[row_idx], "exit_price"] = flat_price
                prop_trades.loc[prop_trades.index[row_idx], "exit_time"] = cutoff_utc
        local_exits = pd.to_datetime(prop_trades.exit_time, utc=True).dt.tz_convert(timezone)
        qty = prop_trades.quantity.astype(float).to_numpy()
        entry = prop_trades.entry_price.astype(float).to_numpy()
        exit_ = prop_trades.exit_price.astype(float).to_numpy()
        side = prop_trades.side.astype(float).to_numpy()
        if pnl_basis == "current_notional":
            gross = ((exit_ - entry) / entry * side) * latest_price * float(point_value) * qty
        else:
            gross = (exit_ - entry) * side * float(point_value) * qty
        if commission_rate_per_side > 0:
            fees = ((np.abs(entry) + np.abs(exit_)) * float(point_value) * qty * commission_rate_per_side)
        else:
            fees = 2.0 * float(commission) * qty
        scaled_pnl = gross - fees
        daily = pd.Series(scaled_pnl, index=local_exits.dt.normalize()).groupby(level=0).sum()
        start, end = daily.index.min(), daily.index.max()
        dates = pd.date_range(start.tz_localize(None), end.tz_localize(None), freq="B").tz_localize(timezone)
        daily = daily.reindex(dates, fill_value=0.0)
        daily_values = daily.to_numpy(dtype=np.float64)
    else:
        daily_values = np.zeros(1, dtype=np.float64)
    horizon = len(daily_values)
    block_days = min(int(block_days), horizon)
    block_count = int(math.ceil(horizon / block_days))
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, horizon, size=(runs, block_count))
    offsets = np.arange(block_days, dtype=np.int64)
    indices = (starts[:, :, None] + offsets[None, None, :]) % horizon
    sampled = daily_values[indices.reshape(runs, -1)[:, :horizon]]
    return sampled, latest_price


def lucidflex_metrics(daily_pnl_paths, account_size, consistency_pct):
    """Estimate LucidFlex evaluation passes and subsequent payout eligibility."""
    account_rules = {
        25000: {"target": 1250, "max_loss": 1000, "min_daily_profit": 100, "payout_cap": 1000},
        50000: {"target": 3000, "max_loss": 2000, "min_daily_profit": 150, "payout_cap": 2000},
        100000: {"target": 6000, "max_loss": 3000, "min_daily_profit": 200, "payout_cap": 2500},
        150000: {"target": 9000, "max_loss": 4500, "min_daily_profit": 250, "payout_cap": 3000},
    }



    account_size = int(account_size)
    if account_size not in account_rules:
        raise ValueError("LucidFlex account size must be 25000, 50000, 100000, or 150000")
    if consistency_pct not in (40, 50):
        raise ValueError("LucidFlex evaluation consistency must be 40 or 50 percent")
    rules = account_rules[account_size]
    passes = np.zeros(len(daily_pnl_paths), dtype=bool)
    pass_days = np.full(len(daily_pnl_paths), -1, dtype=np.int32)
    payout_counts = np.zeros(len(daily_pnl_paths), dtype=np.int8)
    net_payouts = np.zeros(len(daily_pnl_paths), dtype=np.float64)
    first_payout_days = np.full(len(daily_pnl_paths), -1, dtype=np.int32)

    for run_idx, daily_pnls in enumerate(daily_pnl_paths):
        balance = float(account_size)
        high_close = balance
        best_day = 0.0
        pass_idx = -1
        for day_idx, daily_pnl in enumerate(daily_pnls):
            balance += float(daily_pnl)
            day_pnl = float(daily_pnl)
            best_day = max(best_day, day_pnl)
            high_close = max(high_close, balance)
            locked_mll = account_size + 100
            mll = max(account_size - rules["max_loss"], min(high_close - rules["max_loss"], locked_mll))
            if balance <= mll:
                break
            profit = balance - account_size
            if profit >= rules["target"] and best_day <= profit * consistency_pct / 100.0:
                passes[run_idx] = True
                pass_days[run_idx] = day_idx + 1
                pass_idx = day_idx
                break

        if pass_idx < 0:
            continue

        # Funded phase starts at the nominal account size. Each payout cycle
        # needs five qualifying green days and positive cycle profit.
        funded_balance = float(account_size)
        funded_high_close = funded_balance
        cycle_profit = 0.0
        qualifying_days = 0
        count = 0
        paid = 0.0
        floor_locked = False
        for day_offset, daily_pnl in enumerate(daily_pnls[pass_idx + 1:], start=pass_idx + 2):
            funded_balance += float(daily_pnl)
            day_pnl = float(daily_pnl)
            cycle_profit += day_pnl
            if day_pnl >= rules["min_daily_profit"]:
                qualifying_days += 1
            funded_high_close = max(funded_high_close, funded_balance)
            lock_balance = account_size + 100
            mll = lock_balance if floor_locked else max(
                account_size - rules["max_loss"],
                min(funded_high_close - rules["max_loss"], lock_balance),
            )
            if funded_balance <= mll:
                break
            if qualifying_days >= 5 and cycle_profit > 0 and count < 5:
                available = max(0.0, funded_balance - lock_balance - 0.01)
                gross_request = min(cycle_profit * 0.5, rules["payout_cap"], available)
                if gross_request >= 500:
                    funded_balance -= gross_request
                    paid += gross_request * 0.90
                    count += 1
                    floor_locked = True
                    qualifying_days = 0
                    cycle_profit = 0.0
                    if first_payout_days[run_idx] < 0:
                        first_payout_days[run_idx] = day_offset
                    if count >= 5:
                        break
        payout_counts[run_idx] = count
        net_payouts[run_idx] = paid

    passed = passes
    eligible = passed & (payout_counts > 0)
    return {
        "provider": "LucidFlex",
        "account_size": account_size,
        "daily_loss_limit_assumption": "off",
        "daily_pnl_basis": "fixed-contract dollar P&L revalued to the latest dataset price; no account-level compounding",
        "flat_by_et": "4:45 PM America/New_York; uses the cutoff bar open when available, otherwise the preceding bar close",
        "drawdown_model": "end-of-day trailing max loss, locking at starting balance + $100",
        "evaluation_profit_target": rules["target"],
        "evaluation_max_loss_limit": rules["max_loss"],
        "evaluation_consistency_limit_pct": consistency_pct,
        "evaluation_pass_rate_pct": float(passes.mean() * 100) if len(passes) else 0.0,
        "evaluation_passed_simulations": int(passes.sum()),
        "mean_days_to_pass": float(pass_days[passed].mean()) if passed.any() else None,
        "payout_rate_pct_of_passed_evaluations": float(eligible.sum() / passed.sum() * 100) if passed.any() else None,
        "mean_payouts_per_passed_evaluation": float(payout_counts[passed].mean()) if passed.any() else None,
        "mean_trader_payout_per_passed_evaluation": float(net_payouts[passed].mean()) if passed.any() else None,
        "mean_trader_payout_when_eligible": float(net_payouts[eligible].mean()) if eligible.any() else None,
        "mean_days_to_first_payout_when_eligible": float(first_payout_days[eligible].mean()) if eligible.any() else None,
        "payout_split_pct_to_trader": 90,
        "profitable_days_required_per_payout": 5,
        "minimum_profit_per_day": rules["min_daily_profit"],
        "maximum_payout_requests": 5,
        "payout_model": "50% of positive cycle profit, capped by account size; $500 minimum; 90/10 split",
    }
def lucidpro_metrics(daily_pnl_paths, account_size, daily_loss_limit_on=False,
                     max_contracts_allowed=None, largest_trade_contracts=0):
    """Estimate LucidPro evaluation passes and payout eligibility under its current published rules."""
    rules_by_size = {
        25000: {"target": 1250, "max_loss": 1000, "dll": None, "payout_goal": 250,
                "payout_first": 1000, "payout_later": 1500},
        50000: {"target": 3000, "max_loss": 2000, "dll": 1200, "payout_goal": 500,
                "payout_first": 2000, "payout_later": 2500},
        100000: {"target": 6000, "max_loss": 3000, "dll": 1800, "payout_goal": 750,
                 "payout_first": 2500, "payout_later": 3000},
        150000: {"target": 9000, "max_loss": 4500, "dll": 2700, "payout_goal": 1000,
                 "payout_first": 3000, "payout_later": 3500},
    }
    account_size = int(account_size)
    if account_size not in rules_by_size:
        raise ValueError("LucidPro account size must be 25000, 50000, 100000, or 150000")
    rules = rules_by_size[account_size]
    contract_ok = (max_contracts_allowed is None or
                   largest_trade_contracts <= max_contracts_allowed)
    passes = np.zeros(len(daily_pnl_paths), dtype=bool)
    pass_days = np.full(len(daily_pnl_paths), -1, dtype=np.int32)
    payout_counts = np.zeros(len(daily_pnl_paths), dtype=np.int8)
    trader_payouts = np.zeros(len(daily_pnl_paths), dtype=np.float64)
    first_payout_days = np.full(len(daily_pnl_paths), -1, dtype=np.int32)
    dll_days = np.zeros(len(daily_pnl_paths), dtype=np.int32)
    payout_limits = []

    for run_idx, raw_path in enumerate(daily_pnl_paths):
        path = np.asarray(raw_path, dtype=np.float64)
        balance = float(account_size)
        peak_close = balance
        pass_idx = -1
        for day_idx, raw_pnl in enumerate(path):
            pnl = float(raw_pnl)
            if daily_loss_limit_on and rules["dll"] and pnl <= -rules["dll"]:
                dll_days[run_idx] += 1
                # Daily results are aggregated. Approximate the optional soft
                # lock by stopping realized losses at the DLL for that date.
                pnl = max(pnl, -float(rules["dll"]))
            balance += pnl
            peak_close = max(peak_close, balance)
            trail_lock = account_size + 100
            mll = max(account_size - rules["max_loss"],
                      min(peak_close - rules["max_loss"], trail_lock))
            if balance <= mll:
                break
            if balance >= account_size + rules["target"]:
                passes[run_idx] = True
                pass_days[run_idx] = day_idx + 1
                pass_idx = day_idx
                break
        if pass_idx < 0 or not contract_ok:
            passes[run_idx] = False
            continue

        funded_balance = float(account_size)
        funded_peak = funded_balance
        cycle_profit = 0.0
        cycle_best_day = 0.0
        paid = 0.0
        count = 0
        for day_offset, raw_pnl in enumerate(path[pass_idx + 1:], start=pass_idx + 2):
            pnl = float(raw_pnl)
            if daily_loss_limit_on and rules["dll"] and pnl <= -rules["dll"]:
                pnl = max(pnl, -float(rules["dll"]))
            funded_balance += pnl
            funded_peak = max(funded_peak, funded_balance)
            mll = max(account_size - rules["max_loss"],
                      min(funded_peak - rules["max_loss"], account_size + 100))
            if funded_balance <= mll:
                break
            cycle_profit += pnl
            cycle_best_day = max(cycle_best_day, pnl)
            if (cycle_profit < rules["payout_goal"] or cycle_profit <= 0 or
                    cycle_best_day > cycle_profit * 0.40):
                continue
            buffer_balance = account_size + rules["max_loss"] + 100
            available = max(0.0, funded_balance - buffer_balance)
            cap = rules["payout_first"] if count == 0 else rules["payout_later"]
            payout_limits.append(cap)
            gross = min(available, float(cap))
            if gross < 500:
                continue
            funded_balance -= gross
            paid += gross * 0.90
            count += 1
            cycle_profit = 0.0
            cycle_best_day = 0.0
            if first_payout_days[run_idx] < 0:
                first_payout_days[run_idx] = day_offset
            if count >= 5:
                break
        payout_counts[run_idx] = count
        trader_payouts[run_idx] = paid

    passed = passes
    eligible = passed & (payout_counts > 0)
    return {
        "provider": "LucidPro",
        "account_size": account_size,
        "evaluation_profit_target": rules["target"],
        "max_loss_limit": rules["max_loss"],
        "daily_loss_limit_enabled": bool(daily_loss_limit_on),
        "daily_loss_limit": rules["dll"] if daily_loss_limit_on else None,
        "daily_loss_limit_model": "soft daily lock; aggregated daily loss is capped at the published limit (approximation)" if daily_loss_limit_on else "off",
        "end_of_day_trailing_mll": True,
        "mll_locks_at_balance": account_size + 100,
        "evaluation_consistency_rule": "none in the published LucidPro evaluation rules",
        "evaluation_pass_rate_pct": float(passed.mean() * 100) if len(passed) else 0.0,
        "evaluation_passed_simulations": int(passed.sum()),
        "mean_days_to_pass": float(pass_days[passed].mean()) if passed.any() else None,
        "max_contracts_allowed": max_contracts_allowed,
        "largest_strategy_order_contracts": largest_trade_contracts,
        "contract_limit_compliant": bool(contract_ok),
        "payout_eligibility_rate_of_passed_evaluations_pct": float(eligible.sum() / passed.sum() * 100) if passed.any() else None,
        "mean_payouts_per_passed_evaluation": float(payout_counts[passed].mean()) if passed.any() else None,
        "mean_trader_payout_per_passed_evaluation": float(trader_payouts[passed].mean()) if passed.any() else None,
        "mean_trader_payout_when_eligible": float(trader_payouts[eligible].mean()) if eligible.any() else None,
        "mean_days_to_first_payout_when_eligible": float(first_payout_days[eligible].mean()) if eligible.any() else None,
        "payout_consistency_pct": 40,
        "payout_profit_goals_per_cycle": rules["payout_goal"],
        "payout_cap_first": rules["payout_first"],
        "payout_cap_later": rules["payout_later"],
        "profit_split_to_trader_pct": 90,
        "flat_by_et": "4:45 PM America/New_York",
        "daily_pnl_basis": "fixed-contract P&L revalued at the latest price; no balance compounding",
        "payout_model": "maximum five payout cycles modeled; 40% best-day consistency, buffer maintained, $500 minimum",
        "rule_sources": [
            "https://support.lucidtrading.com/en/articles/12890029-lucidpro-evaluation-account",
            "https://support.lucidtrading.com/en/articles/12890092-lucidpro-payouts",
            "https://support.lucidtrading.com/en/articles/12890136-lucidpro-drawdown",
            "https://support.lucidtrading.com/en/articles/12890122-lucidpro-daily-loss-limit",
            "https://support.lucidtrading.com/en/articles/11404729-allowed-trading-times",
        ],
    }


CRYPTO_PROP_ACCOUNT_SIZES = (5_000, 10_000, 25_000, 50_000, 100_000)


def make_crypto_prop_pnl_paths(trades, runs, seed, block_days=5):
    """Build UTC-calendar daily net P&L paths for 24/7 crypto prop simulations."""
    if runs < 1 or block_days < 1:
        raise ValueError("crypto prop Monte Carlo runs and block length must be positive")
    if trades.empty:
        daily_values = np.zeros(1, dtype=np.float64)
    else:
        exit_days = pd.to_datetime(trades.exit_time, utc=True).dt.floor("D")
        daily = trades.assign(_exit_day=exit_days).groupby("_exit_day").pnl.sum()
        calendar = pd.date_range(daily.index.min(), daily.index.max(), freq="D")
        daily_values = daily.reindex(calendar, fill_value=0.0).to_numpy(dtype=np.float64)
    horizon = max(1, len(daily_values))
    block_days = min(int(block_days), horizon)
    block_count = int(math.ceil(horizon / block_days))
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, horizon, size=(runs, block_count))
    offsets = np.arange(block_days, dtype=np.int64)
    indices = (starts[:, :, None] + offsets[None, None, :]) % horizon
    daily_paths = daily_values[indices.reshape(runs, -1)[:, :horizon]]
    return daily_paths, horizon


def crypto_prop_metrics(daily_pnl_paths, rule_config):
    """Estimate challenge pass/breach outcomes for five synthetic average-rule tiers."""
    target_pct = float(rule_config["profit_target_pct"])
    max_dd_pct = float(rule_config["maximum_drawdown_pct"])
    daily_dd_pct = float(rule_config["daily_drawdown_pct"])
    if min(target_pct, max_dd_pct, daily_dd_pct) <= 0:
        raise ValueError("crypto prop target and drawdown percentages must be positive")
    runs, horizon = daily_pnl_paths.shape
    accounts = {}
    total = len(CRYPTO_PROP_ACCOUNT_SIZES) * runs
    interval = max(1, int(math.ceil(total / 20)))
    for account_size_index, account_size in enumerate(CRYPTO_PROP_ACCOUNT_SIZES):
        target = account_size * target_pct / 100.0
        max_loss = account_size * max_dd_pct / 100.0
        daily_loss = account_size * daily_dd_pct / 100.0
        pass_days = []
        breach_days = []
        unresolved = 0
        for run_idx, pnl_path in enumerate(daily_pnl_paths):
            balance = float(account_size)
            floor = account_size - max_loss
            passed = False
            breached = False
            for day_idx, day_pnl in enumerate(pnl_path, start=1):
                # Trades are net of fees/slippage. Daily aggregation uses UTC
                # close dates; floating intraday excursions cannot be inferred
                # from a closed-trade series alone.
                if day_pnl <= -daily_loss or balance + day_pnl <= floor:
                    breached = True
                    breach_days.append(day_idx)
                    break
                balance += float(day_pnl)
                if balance >= account_size + target:
                    passed = True
                    pass_days.append(day_idx)
                    break
            if not passed and not breached:
                unresolved += 1
            completed = account_size_index * runs + run_idx + 1
            if completed % interval == 0 or completed == total:
                progress = completed * 100.0 / total
                print(f"Crypto prop progress: {progress:.0f}% ({account_size:,} account tier)",
                      file=sys.stderr, flush=True)
        accounts[str(account_size)] = {
            "account_size": account_size,
            "profit_target": round(target, 3),
            "profit_target_pct": target_pct,
            "maximum_drawdown_limit": round(max_loss, 3),
            "maximum_drawdown_pct": max_dd_pct,
            "static_equity_floor": round(floor, 3),
            "daily_drawdown_limit": round(daily_loss, 3),
            "daily_drawdown_pct": daily_dd_pct,
            "evaluation_pass_rate_pct": round(len(pass_days) * 100.0 / runs, 3) if runs else 0.0,
            "drawdown_breach_rate_pct": round(len(breach_days) * 100.0 / runs, 3) if runs else 0.0,
            "not_passed_or_breached_by_history_end_pct": round(unresolved * 100.0 / runs, 3) if runs else 0.0,
            "mean_days_to_pass": round(float(np.mean(pass_days)), 3) if pass_days else None,
            "mean_days_to_breach": round(float(np.mean(breach_days)), 3) if breach_days else None,
            "simulations": int(runs),
            "history_horizon_calendar_days": int(horizon),
        }
    return {
        "provider": "Synthetic average of published crypto prop challenge rules",
        "pnl_basis": "actual realized net trade P&L at selected strategy position size; no balance compounding",
        "evaluation_day": "UTC calendar day (24/7 crypto calendar)",
        "risk_model_limitations": "Daily net closed P&L approximation; intraday floating equity, individual firm's trailing rules, consistency rules, and minimum trading days are not averaged.",
        "rule_providers": rule_config.get("rule_providers", []),
        "rule_sources": rule_config.get("rule_sources", []),
        "accounts": accounts,
    }


def _ask(prompt: str, default: str | None = None) -> str:
    suffix = f" [{default}]" if default else ""
    value = input(f"{prompt}{suffix}: ").strip()
    return value or (default or "")


def _parse_assets(value: str, parser) -> list[str]:
    assets = list(dict.fromkeys(part.strip().upper() for part in value.split(",") if part.strip()))
    invalid = [asset for asset in assets if asset not in {"NQ", "ES", "GC", "BTC", "MNQ", "MES", "MGC", "CL", "ZN"}]
    if not assets or invalid:
        parser.error("asset must be one or more comma-separated choices: NQ, ES, GC, BTC, MNQ, MES, MGC, CL, ZN")
    return assets


def _resolve_strategy(value: str, parser) -> Path:
    raw = Path(value.strip().strip('"'))
    candidate = raw if raw.is_absolute() else (Path.cwd() / raw if (Path.cwd() / raw).exists() else ROOT / raw)
    candidate = candidate.resolve()
    if not candidate.is_file():
        parser.error(f"strategy file not found: {candidate}")
    if candidate.suffix.lower() not in {".pl", ".pine", ".pinescript"}:
        parser.error("strategy must be a .pl, .pine, or .pinescript file")
    if candidate.suffix.lower() in {".pine", ".pinescript"}:
        parser.error("PineScript files are recognized, but this engine currently executes PowerLanguage only; Pine execution needs its own interpreter and is not silently converted.")
    return candidate


def _run_asset(args, parser, cfg, asset, strategy_source):
    asset_run_started = time.monotonic()
    try:
        input_defaults, declared_variables, parsed_tree = parse_source(strategy_source)
        compiled = compile_program(parsed_tree, set(input_defaults) | set(declared_variables))
        required_names = {name.lower() for name in compiled.co_names}
    except PowerLanguageError as exc:
        parser.error(f"strategy parse error: {exc}")
    normalized_required_names = {INDICATOR_ALIASES.get(name, name) for name in required_names}
    market_depth_studies = {
        "marketdepthbidprice", "marketdepthaskprice", "marketdepthbidsize", "marketdepthasksize",
        "marketdepthbid", "marketdepthask", "marketdepthimbalance", "marketdepthspread", "marketdepthmicroprice",
        "marketdepthbboimbalance", "marketdepthtopimbalance", "marketdepthweightedmid", "marketdepthweightedmidprice",
    }.intersection(required_names | normalized_required_names)
    quote_studies = {
        "bidaskspread", "spread", "bidaskmidpoint", "quoteprice", "bidaskvolratio", "bidaskvolumeratio",
        "bidandaskvolratio",
    }.intersection(required_names | normalized_required_names)
    uses_spread_series = bool(normalized_required_names & {"spreadratio", "spreaddiff"})
    needs_mbo = any(name.startswith("mbo_") for name in required_names) or bool(market_depth_studies)
    external_data2_required = bool(re.search(
        r"(?im)^\s*//\s*ENGINE_DATA2_EXTERNAL_REQUIRED\s*=\s*true\s*$", strategy_source
    ))
    if external_data2_required and args.data2 is None:
        parser.error("this strategy requires a separate Data2 file; provide an implied-volatility OHLCV dataset with --data2")
    needs_flow = any(name.startswith("flow_") for name in required_names)
    needs_ofb = any(name.startswith("ofb_") for name in required_names)
    needs_feed_quality = any(name.startswith("feed_") for name in required_names)
    tick_names = {"upticks", "downticks"} & required_names
    flow_source_default = input_defaults.get("flowsource")
    needs_tick_data = bool(tick_names) and (flow_source_default is None or int(flow_source_default) == 2)
    needs_quotes = bool(required_names & {"bid", "ask", "bidsize", "asksize", "insidebid", "insideask",
                                          "bid_at", "ask_at", "bidsize_at", "asksize_at"})
    needs_quotes = needs_quotes or bool(quote_studies)
    needs_depth = bool(market_depth_studies) or any(
        name.startswith("mbo_bid_price_") or name.startswith("mbo_ask_price_") or
        name.startswith("mbo_bid_size_") or name.startswith("mbo_ask_size_") or
        name in {"mbo_depth_imbalance_5", "mbo_microprice", "mbo_spread", "mbo_bid_depth_5", "mbo_ask_depth_5"}
        for name in required_names
    )
    imported_data = None
    if args.data is None:
        data_candidates = []
        imported_tbbo = ROOT / "data" / "imported_20261030" / "futures_tbbo" / asset
        imported_tbbo_available = imported_tbbo.is_dir() and any(imported_tbbo.glob("*.parquet"))
        requires_full_depth = any(name.startswith(("mbo_bid_price_", "mbo_ask_price_", "mbo_bid_size_", "mbo_ask_size_"))
                                  for name in required_names)
        legacy_nq_mbo = ROOT / "data" / "NQ" / "raw" / "orderbook" / "daily_events"
        if asset in {"MNQ", "MES", "CL", "ZN"} and imported_tbbo_available:
            imported_data = _imported_tbbo_bars(asset)
        elif asset == "MGC" and needs_mbo:
            # The MGC book dates do not overlap the available MGC trade feed.
            # Use its midpoint as an explicitly labeled quote-price proxy.
            imported_data = _imported_mgc_bbo_bars()
        elif needs_mbo or needs_flow or needs_quotes or needs_tick_data or needs_feed_quality:
            if imported_tbbo_available:
                imported_data = _imported_tbbo_bars(asset)
                if asset == "NQ" and needs_mbo and legacy_nq_mbo.is_dir():
                    imported_data = None
        if needs_mbo or needs_flow or needs_depth or needs_tick_data or needs_feed_quality:
            if asset == "NQ":
                event_dir = ROOT / "data" / "NQ" / "raw" / "orderbook" / "daily_events"
                if event_dir.exists():
                    data_candidates.append(event_dir)
            tbbo = sorted((ROOT / "data" / "raw" / "tbbo").glob(f"{asset}_*_tbbo.dbn"))
            data_candidates.extend(tbbo)
        if needs_quotes:
            data_candidates.extend(sorted((ROOT / "data" / "raw" / "tbbo").glob(f"{asset}_*_tbbo.dbn")))
        data_candidates += [ROOT / "data" / f"{asset}{ext}" for ext in (".parquet", ".csv", ".csv.gz")]
        data_candidates.append(ROOT / "data" / "OFB" / f"ofb_bars_{asset}.csv")
        nested_parquet = ROOT / "data" / asset / "ohlcv" / f"{asset.lower()}_1m_full.parquet"
        data_candidates.append(nested_parquet)
        data_candidates += sorted((ROOT / "data" / "raw" / "tbbo").glob(f"{asset}_*_tbbo.dbn"))
        data_path = next((candidate for candidate in data_candidates if candidate.exists()), None)
        if asset == "NQ" and needs_mbo and legacy_nq_mbo.is_dir():
            data_path = legacy_nq_mbo
        if imported_data is not None:
            data_path = (ROOT / "data" / "imported_20261030" / "l2_book" / "MGC" / "mgc_bbo_midpoint_1min_v3.parquet"
                         if asset == "MGC" else
                     _imported_tbbo_cache_path(asset))
        # Bridge functions must use the synchronized OFB bars and sidecars by
        # default; ordinary NQ scripts retain the longer root NQ history.
        ofb_bars = ROOT / "data" / "OFB" / f"ofb_bars_{asset}.csv"
        if needs_ofb and asset == "NQ" and ofb_bars.is_file():
            imported_data = None
            data_path = ofb_bars
    elif "{asset}" in str(args.data):
        data_path = Path(str(args.data).format(asset=asset))
    elif args.data.is_dir():
        direct = [args.data / f"{asset}{ext}" for ext in (".parquet", ".csv", ".csv.gz", ".dbn")]
        direct.append(args.data / f"{asset.lower()}_1m_full.parquet")
        direct += [args.data / "OFB" / f"ofb_bars_{asset}.csv", args.data / f"ofb_bars_{asset}.csv"]
        data_path = next((candidate for candidate in direct if candidate.exists()), None)
        if data_path is None and (list(args.data.glob("*.csv.gz")) or list(args.data.glob("*.csv"))):
            data_path = args.data
    else:
        data_path = args.data
    if data_path is None or not data_path.exists():
        parser.error(f"no {asset} dataset found; expected data/{asset}.csv, .parquet, .csv.gz, or .dbn")
    print(f"\n[{asset}] Loading data from {data_path}...", file=sys.stderr, flush=True)
    inferred = infer_strategy_settings(strategy_source)
    data_timezone = inferred.get("data_timezone", "UTC")
    if data_path.name.lower().startswith("ofb_bars_") and "data_timezone" not in inferred:
        data_timezone = "America/New_York"
    data = imported_data if imported_data is not None else load_data(data_path, data_timezone, needs_depth)
    if any(name.startswith("ofb_") for name in required_names):
        ofb_folder = data_path.parent / "OFB"
        if not (ofb_folder / "ofb_hist.bin").is_file():
            ofb_folder = data_path.parent
        if any((ofb_folder / name).is_file() for name in ("ofb_hist.bin", "ofb_gex.bin", "ofb_cot.bin")):
            try:
                data = attach_ofb_features(data, ofb_folder, strategy_source, timezone=data_timezone)
            except (ValueError, OSError) as exc:
                parser.error(f"could not attach OrderFlowBridge data: {exc}")
        elif needs_ofb:
            parser.error("this strategy uses OFB functions, but no ofb_*.bin sidecar files were found beside the selected data")
    if quote_studies.intersection({"bidaskvolratio", "bidaskvolumeratio", "bidandaskvolratio"}):
        needed_quote_columns={"bid_size","ask_size"}
    else:
        needed_quote_columns={"bid","ask"} if quote_studies else set()
    if needed_quote_columns and not needed_quote_columns.issubset(data.columns):
        parser.error(
            f"{asset} selected data lacks fields required by the quote study: "
            + ", ".join(sorted(needed_quote_columns-set(data.columns)))
        )
    if market_depth_studies:
        required_book_fields = {"mbo_bid_size_1", "mbo_ask_size_1"}
        if asset != "NQ" or not required_book_fields.issubset(data.columns):
            parser.error(
                "MarketDepth studies require synchronized NQ L2/L3 book snapshots; "
                "this dataset/asset does not contain the needed book fields"
            )
    data2_path = None
    data2_frame = None
    if args.data2 is not None:
        if "{asset}" in str(args.data2):
            data2_path = Path(str(args.data2).format(asset=asset))
        elif args.data2.is_dir():
            data2_candidates = [args.data2 / f"{asset}{ext}" for ext in (".parquet", ".csv", ".csv.gz")]
            data2_path = next((candidate for candidate in data2_candidates if candidate.is_file()), None)
        else:
            data2_path = args.data2
        if data2_path is None or not data2_path.is_file():
            parser.error(f"no Data2 dataset found for {asset}; provide a file or an asset-keyed directory")
        try:
            data2_frame = load_data(data2_path, data_timezone)
            data2_frame = limit_data_period(data2_frame, args.data_period)
        except (ValueError, OSError) as exc:
            parser.error(f"could not load Data2 dataset {data2_path}: {exc}")
    if asset == "MGC" and needs_mbo and imported_data is None:
        data = _attach_mgc_book(data)
    quote_midpoint_proxy = bool(data.get("data_is_quote_midpoint_proxy", pd.Series(False, index=data.index)).fillna(False).any())
    if quote_midpoint_proxy:
        print(f"[{asset}] Using midpoint-derived quote bars (no executed volume/trade OHLC in this source).",
              file=sys.stderr, flush=True)
    if needs_mbo:
        required_mbo = sorted(name for name in required_names if name.startswith("mbo_"))
        if required_mbo:
            present = [name for name in required_mbo if name in data.columns]
            if len(present) != len(required_mbo):
                absent = sorted(set(required_mbo) - set(present))
                parser.error(f"{asset} data does not provide these requested order-book fields: {', '.join(absent)}")
            valid_book = data[present].notna().all(axis=1)
            data = data.loc[valid_book].copy()
            if data.empty:
                parser.error(f"{asset} has no overlapping bars for the requested order-book features")
            print(f"[{asset}] Restricting order-book strategy to {len(data):,} bars with synchronized {asset} book data.",
                  file=sys.stderr, flush=True)
    all_source_bars = len(data)
    try:
        data = limit_data_period(data, args.data_period)
    except ValueError as exc:
        parser.error(str(exc))
    input_bars = len(data)
    try:
        offset = parse_data_period(args.data_period)
    except ValueError:
        offset = None
    if offset is not None and not data.empty:
        requested_start = data.timestamp.iloc[-1] - offset
        if data.timestamp.iloc[0] > requested_start:
            actual_days = int((data.timestamp.iloc[-1] - data.timestamp.iloc[0]).total_seconds() // 86400) + 1
            print(f"[{asset}] Requested {args.data_period}; this source has about {actual_days} days. "
                  "Using all available data.", file=sys.stderr, flush=True)
    if args.data1_session:
        try:
            data = filter_session(data, args.data1_session, args.session_timezone or inferred.get("session_timezone", "UTC"))
        except ValueError as exc:
            parser.error(str(exc))
    try:
        data = resample_primary(data, args.bar_timeframe)
    except Exception as exc:
        parser.error(f"invalid primary bar timeframe {args.bar_timeframe!r}: {exc}")
    print(f"[{asset}] {input_bars:,} source bars selected from {all_source_bars:,}; {len(data):,} strategy bars at {args.bar_timeframe} "
          f"({data.timestamp.iloc[0]} to {data.timestamp.iloc[-1]}). Starting...", file=sys.stderr, flush=True)
    data_preparation_seconds = time.monotonic() - asset_run_started
    runtime_estimate_seconds, runtime_remaining_seconds, runtime_basis = _estimate_runtime(
        asset, args.strategy.stem, args.bar_timeframe, len(data), data_preparation_seconds)
    print(f"[{asset}] Runtime estimate: total_seconds={runtime_estimate_seconds:.0f} "
          f"remaining_seconds={runtime_remaining_seconds:.0f}; basis={runtime_basis}",
          file=sys.stderr, flush=True)
    cash = args.initial_cash if args.initial_cash is not None else cfg["initial_cash"]
    asset_commission = cfg["assets"][asset].get("commission_per_contract", cfg["commission_per_contract"])
    commission = args.commission if args.commission is not None else asset_commission
    commission_rate_per_side = (
        0.0 if args.commission is not None else
        float(cfg["assets"][asset].get("commission_rate_per_side", 0.0))
    )
    slip = args.slippage_ticks if args.slippage_ticks is not None else cfg["slippage_ticks"]
    timezone = args.session_timezone or cfg.get("session_timezone", "UTC")
    strategy_started = time.monotonic()
    try:
        trades, equity, inputs = run_powerlanguage(data, strategy_source, cfg["assets"][asset], cash, commission, slip,
                                                  args.progress_interval, args.data2_timeframe, timezone,
                                                  args.num_contracts, commission_rate_per_side, data2_frame)
    except PowerLanguageError as exc:
        parser.error(f"{asset}: {exc}")
    strategy_runtime_seconds = time.monotonic() - strategy_started
    print(f"[{asset}] Backtest complete; calculating metrics and simulations...", file=sys.stderr, flush=True)
    point_value = float(cfg["assets"][asset]["point_value"])
    if len(trades):
        price_returns = (trades.exit_price.astype(float) - trades.entry_price.astype(float)) / trades.entry_price.astype(float) * trades.side.astype(float)
        trades["price_return_pct"] = price_returns * 100.0
        trade_notional = trades.entry_price.astype(float) * point_value * trades.quantity.astype(float)
        trades["net_notional_return_pct"] = trades.pnl.astype(float) / trade_notional * 100.0
    else:
        trades["price_return_pct"] = pd.Series(dtype=float)
        trades["net_notional_return_pct"] = pd.Series(dtype=float)
    stats = calc_stats(trades, equity, cash, timezone)
    if quote_midpoint_proxy:
        stats["data_quality_notes"] = ["MGC OHLC is a midpoint-of-book proxy, not executed trade OHLC; volume is unavailable."]
    stats["num_contracts_entry_override"] = args.num_contracts
    stats["data_period_requested"] = args.data_period
    stats["source_bars_in_selected_period"] = input_bars
    stats["source_bars_available"] = all_source_bars
    if data2_path is not None:
        stats["data2_source"] = str(data2_path)
        stats["data2_source_bars"] = len(data2_frame)
    stats["strategy_timeframe"] = args.bar_timeframe
    stats["data_preparation_seconds"] = data_preparation_seconds
    stats["strategy_execution_seconds"] = strategy_runtime_seconds
    stats["commission_per_contract_per_side"] = commission if commission_rate_per_side == 0 else None
    stats["commission_rate_per_side_pct"] = commission_rate_per_side * 100.0
    stats["commission_model"] = (
        f"{commission_rate_per_side * 100:g}% of notional per side" if commission_rate_per_side > 0
        else "fixed currency amount per contract per side"
    )
    stats["slippage_ticks_per_side"] = slip
    buy_hold = data.close.astype(float) / float(data.close.iloc[0])
    stats["buy_and_hold_benchmark"] = {
        "return_pct": float((buy_hold.iloc[-1] - 1.0) * 100.0),
        "max_drawdown_pct": float(((buy_hold / buy_hold.cummax()) - 1.0).min() * 100.0),
        "method": "buy and hold from first to last close in selected data window",
    }
    mc, terminal, drawdowns, mc_bands, mc_samples, mc_paths = monte_carlo(
        equity, cash, int(cfg["monte_carlo_runs"]), int(cfg["random_seed"]),
        int(cfg.get("monte_carlo_block_days", 5)), int(cfg.get("monte_carlo_sample_paths", 20)))
    stats["monte_carlo"] = mc
    if asset == "BTC":
        stats["lucidflex"] = {"status": "not_applicable", "reason": "LucidFlex does not offer crypto products."}
        stats["lucidpro"] = {"status": "not_applicable", "reason": "LucidPro does not offer crypto products."}
        crypto_paths, crypto_horizon = make_crypto_prop_pnl_paths(
            trades, int(cfg["monte_carlo_runs"]), int(cfg["random_seed"]),
            int(cfg.get("monte_carlo_block_days", 5)))
        stats["crypto_prop"] = crypto_prop_metrics(crypto_paths, cfg["crypto_prop_average"])
        stats["crypto_prop"]["history_horizon_calendar_days"] = int(crypto_horizon)
    else:
        lucid_cfg = cfg.get("lucidflex", {})
        account_size = int(args.lucidflex_account or lucid_cfg.get("account_size", 100000))
        pnl_basis = getattr(args, "lucid_pnl_basis", "points")
        lucid_pnl_paths, current_price = make_lucidflex_pnl_paths(
            trades, data, point_value, commission, account_size, timezone, args.bar_timeframe,
            int(cfg["monte_carlo_runs"]), int(cfg["random_seed"]),
            int(cfg.get("monte_carlo_block_days", 5)), commission_rate_per_side, pnl_basis)
        lucid_stats = lucidflex_metrics(lucid_pnl_paths, account_size,
                                        int(lucid_cfg.get("consistency_pct", 40)))
        lucid_stats["current_reference_price"] = current_price
        lucid_stats["point_value_per_contract"] = point_value
        lucid_stats["selected_account_only"] = True
        lucid_stats["pnl_basis"] = pnl_basis
        stats["lucidflex"] = lucid_stats
        pro_account_size = int(args.lucidpro_account)
        mini_cap_by_size = {25000: 2, 50000: 4, 100000: 6, 150000: 10}
        allowed_contracts = mini_cap_by_size[pro_account_size] * (10 if asset in {"MNQ", "MES", "MGC"} else 1)
        largest_trade_qty = float(trades.quantity.max()) if len(trades) else 0.0
        pro_stats = lucidpro_metrics(lucid_pnl_paths, pro_account_size, args.lucidpro_dll,
                                    allowed_contracts, largest_trade_qty)
        pro_stats["current_reference_price"] = current_price
        pro_stats["point_value_per_contract"] = point_value
        stats["lucidpro"] = pro_stats
    stats = _round_report_values(stats)
    strategy_dir = re.sub(r"[^A-Za-z0-9._-]+", "_", args.strategy.stem).strip("._") or "strategy"
    run_tag = getattr(args, "run_tag", None)
    if run_tag:
        run_tag = re.sub(r"[^A-Za-z0-9._-]+", "_", str(run_tag)).strip("._")
    out = ROOT / "results" / asset / strategy_dir
    if run_tag:
        out = out / run_tag
    out.mkdir(parents=True, exist_ok=True)
    trades.to_csv(out / "trades.csv", index=False)
    if len(trades):
        normalized = trades[["entry_time", "exit_time", "side", "quantity", "entry_price", "exit_price",
                             "price_return_pct", "net_notional_return_pct"]].copy()
        normalized["cumulative_price_return_index"] = (1.0 + normalized.price_return_pct / 100.0).cumprod()
    else:
        normalized = pd.DataFrame(columns=["entry_time", "exit_time", "side", "quantity", "entry_price", "exit_price",
                                           "price_return_pct", "net_notional_return_pct", "cumulative_price_return_index"])
    normalized.to_csv(out / "normalized_trade_returns.csv", index=False)
    equity.to_csv(out / "equity.csv", index=False)
    mc_bands.to_csv(out / "monte_carlo_equity.csv", index=False)
    mc_samples.to_csv(out / "monte_carlo_sample_paths.csv", index=False)
    (out / "stats.json").write_text(json.dumps(stats, indent=2, default=str), encoding="utf-8")
    plot_stride = max(1, math.ceil(len(equity) / 200_000))
    plot_equity = equity.iloc[::plot_stride]
    fig, ax = plt.subplots(figsize=(12, 5)); ax.plot(plot_equity.timestamp, plot_equity.equity, lw=0.8); ax.set(title=f"{asset} equity curve", ylabel="Equity"); _format_currency_axis(ax); ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(out / "equity_curve.png", dpi=140); plt.close(fig)
    strategy_plot_columns = [column for column in equity.columns if re.fullmatch(r"plot_\d+", str(column))]
    if strategy_plot_columns:
        fig, ax = plt.subplots(figsize=(12, 5))
        plot_times = pd.to_datetime(plot_equity.timestamp, utc=True)
        for column in strategy_plot_columns:
            ax.plot(plot_times, plot_equity[column], lw=0.75, label=f"Plot {column.split('_')[1]}")
        ax.set(title=f"{asset} PowerLanguage plots", ylabel="Strategy plot value")
        ax.grid(alpha=.25)
        ax.legend(loc="best")
        fig.tight_layout(); fig.savefig(out / "strategy_plots.png", dpi=140); plt.close(fig)
    fig, ax = plt.subplots(figsize=(12, 5))
    x = mc_bands.simulation_day
    ax.fill_between(x, mc_bands.equity_p05, mc_bands.equity_p95, alpha=.18, label="5th–95th percentile")
    ax.fill_between(x, mc_bands.equity_p25, mc_bands.equity_p75, alpha=.28, label="25th–75th percentile")
    ax.plot(x, mc_bands.equity_median, lw=1.5, label="Simulated median")
    for col in mc_samples.columns[1:11]: ax.plot(mc_samples.simulation_day, mc_samples[col], lw=.55, alpha=.35, color="gray")
    ax.plot(x, mc_bands.actual_equity, lw=1, ls="--", color="black", label="Actual equity")
    ax.set(title=f"{asset} Monte Carlo equity paths", xlabel="Trading days", ylabel="Equity")
    _format_currency_axis(ax)
    ax.grid(alpha=.25); ax.legend(); fig.tight_layout(); fig.savefig(out / "monte_carlo_equity.png", dpi=140); plt.close(fig)
    if len(normalized):
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(pd.to_datetime(normalized.exit_time, utc=True), normalized.cumulative_price_return_index, lw=1.0)
        ax.set_yscale("log")
        ax.set(title=f"{asset} cumulative price-normalized trade returns", ylabel="Cumulative return index (log scale)")
        ax.grid(alpha=.25)
        fig.tight_layout(); fig.savefig(out / "normalized_trade_returns.png", dpi=140); plt.close(fig)
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    axs[0].hist(terminal, bins=40); axs[0].axvline(cash, color="red", ls="--"); axs[0].set(title="Terminal equity", xlabel="Equity"); _format_currency_axis(axs[0], "x")
    axs[1].hist(drawdowns, bins=40); axs[1].set(title="Maximum drawdown", xlabel="Currency"); _format_currency_axis(axs[1], "x")
    fig.tight_layout(); fig.savefig(out / "monte_carlo.png", dpi=140); plt.close(fig)
    chart_names = ["equity_curve.png", "monte_carlo_equity.png", "monte_carlo.png"]
    if strategy_plot_columns:
        chart_names.append("strategy_plots.png")
    if len(normalized):
        chart_names.append("normalized_trade_returns.png")
    _write_pdf_report(out, asset, args, input_bars, all_source_bars, len(data), inputs, stats, chart_names)
    stats["runtime_seconds"] = time.monotonic() - asset_run_started
    stats["processing_runtime_seconds"] = max(0.0, stats["runtime_seconds"] - data_preparation_seconds)
    stats["runtime_estimate_seconds"] = runtime_estimate_seconds
    stats["runtime_estimate_basis"] = runtime_basis
    stats = _round_report_values(stats)
    (out / "stats.json").write_text(json.dumps(stats, indent=2, default=str), encoding="utf-8")
    stale_html_report = out / "report.html"
    if stale_html_report.exists():
        stale_html_report.unlink()
    print(f"[{asset}] Results saved to {out}", file=sys.stderr, flush=True)
    print(json.dumps({"asset": asset, "result_folder": str(out), "stats": stats}, indent=2, default=str))


def main():
    p = argparse.ArgumentParser(description="Interactive PowerLanguage backtester")
    p.add_argument("--asset", help="one or more comma-separated assets: NQ, ES, GC, BTC, MNQ, MES, MGC, CL, ZN")
    p.add_argument("--strategy", type=Path, help="PowerLanguage strategy file (.pl)")
    p.add_argument("--data", type=Path, help="CSV/parquet file, asset directory, or path containing {asset}")
    p.add_argument("--data2", type=Path, help="Separate Data2 CSV/parquet file or asset-keyed directory; needed for strategies with an external series")
    p.add_argument("--initial-cash", type=float)
    p.add_argument("--commission", type=float)
    p.add_argument("--slippage-ticks", type=float)
    p.add_argument("--progress-interval", type=float, default=3.0)
    p.add_argument("--bar-timeframe", help="strategy interval, e.g. 1min, 5min, 1h, 4h, 1D")
    p.add_argument("--data2-timeframe", help="PowerLanguage Data2 interval (defaults to a strategy's ENGINE_DATA2_TIMEFRAME marker or 5min)")
    p.add_argument("--session-timezone", help="timezone for PowerLanguage Date/Time expressions")
    p.add_argument("--data1-session", help="restrict Data1 to a session such as 0930-1600")
    p.add_argument("--data-period", help="history window such as 4y, 18mo, 90d, or all")
    p.add_argument("--lucidflex-account", help="LucidFlex evaluation size: 25k, 50k, 100k, or 150k")
    p.add_argument("--lucidpro-account", help="LucidPro evaluation size: 25k, 50k, 100k, or 150k")
    p.add_argument("--lucidpro-dll", choices=("on", "off"), help="enable or disable LucidPro's optional daily loss limit")
    p.add_argument("--lucid-pnl-basis", choices=("points", "current_notional"), default="points",
                   help="prop-account P&L model: historical point movement (default) or scaled to latest contract notional")
    p.add_argument("--num-contracts", type=float, help="force each strategy entry to this many contracts (fractions allowed, e.g. 0.1)")
    p.add_argument("--run-tag", help=argparse.SUPPRESS)
    args = p.parse_args()

    asset_text = args.asset or _ask("What asset(s) do you want to backtest? (comma-separated: NQ, ES, GC, BTC, MNQ, MES, MGC, CL, ZN)")
    assets = _parse_assets(asset_text, p)
    args.data_period = args.data_period or _ask("How much data should the backtest use? (e.g. 4y, 18mo, 90d, all)", "all")
    try:
        parse_data_period(args.data_period)
    except ValueError as exc:
        p.error(str(exc))
    strategy_default = "strategy.pl" if (ROOT / "strategy.pl").exists() else None
    strategy_text = str(args.strategy) if args.strategy else _ask("Which strategy file should I run? (.pl)", strategy_default)
    args.strategy = _resolve_strategy(strategy_text, p)
    if len(assets) > 1 and args.data is not None and not args.data.is_dir() and "{asset}" not in str(args.data):
        p.error("for multiple assets, --data must be a directory containing one file per asset or include {asset} in its path")
    if len(assets) > 1 and args.data2 is not None and not args.data2.is_dir() and "{asset}" not in str(args.data2):
        p.error("for multiple assets, --data2 must be a directory containing one file per asset or include {asset} in its path")
    if len(assets) > 1 and args.data is not None and args.data.is_dir() and \
            not any((args.data / f"{asset}{ext}").exists() for asset in assets for ext in (".parquet", ".csv", ".csv.gz", ".dbn")) and \
            (list(args.data.glob("*.csv.gz")) or list(args.data.glob("*.csv"))):
        p.error("a folder of event CSVs represents one instrument; select only that asset or provide separate per-asset files")
    cfg = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    results_root = ROOT / "results"
    for configured_asset in cfg["assets"]:
        (results_root / configured_asset).mkdir(parents=True, exist_ok=True)
    strategy_source = args.strategy.read_text(encoding="utf-8-sig")
    declared_inputs = parse_source(strategy_source)[0]
    num_contracts_default = float(declared_inputs.get("numcontracts", 1))
    if args.num_contracts is None:
        value = _ask("NumContracts: how many contracts should each entry use? (fractions allowed, e.g. 0.1)",
                     f"{num_contracts_default:g}")
        try:
            args.num_contracts = float(value)
        except ValueError:
            p.error("NumContracts must be a positive number (fractions such as 0.1 are allowed)")
    if not math.isfinite(args.num_contracts) or args.num_contracts <= 0:
        p.error("NumContracts must be a positive finite number")
    inferred = infer_strategy_settings(strategy_source)
    if args.bar_timeframe is None:
        timeframe_default = inferred.get("bar_timeframe", "1min")
        args.bar_timeframe = _ask("What timeframe should Data1 use? (e.g. 1min, 5min, 1h, 4h, 1D)", timeframe_default)
    if args.data2_timeframe is None:
        args.data2_timeframe = inferred.get("data2_timeframe", "5min")
    if args.session_timezone is None:
        args.session_timezone = inferred.get("session_timezone")
    if args.data1_session is None:
        args.data1_session = inferred.get("data1_session")
    lucid_cfg = cfg.get("lucidflex", {})
    if args.lucidflex_account is None and any(asset != "BTC" for asset in assets):
        default_account = int(lucid_cfg.get("account_size", 100000))
        account_text = _ask("LucidFlex account size for the evaluation/payout section? (25k, 50k, 100k, 150k)", f"{default_account // 1000}k")
    else:
        account_text = args.lucidflex_account or f"{int(lucid_cfg.get('account_size', 100000)) // 1000}k"
    normalized_account = str(account_text).strip().lower().replace(",", "").replace("$", "")
    if normalized_account.endswith("k"):
        try:
            args.lucidflex_account = int(float(normalized_account[:-1]) * 1000)
        except ValueError:
            p.error("LucidFlex account size must be 25k, 50k, 100k, or 150k")
    else:
        try:
            args.lucidflex_account = int(normalized_account)
        except ValueError:
            p.error("LucidFlex account size must be 25k, 50k, 100k, or 150k")
    if args.lucidflex_account not in {25000, 50000, 100000, 150000}:
        p.error("LucidFlex account size must be 25k, 50k, 100k, or 150k")
    pro_account_text = args.lucidpro_account
    if pro_account_text is None and any(asset != "BTC" for asset in assets):
        pro_default = int(cfg.get("lucidpro", {}).get("account_size", args.lucidflex_account))
        pro_account_text = _ask("LucidPro account size for evaluation/payout stats? (25k, 50k, 100k, 150k)",
                                f"{pro_default // 1000}k")
    pro_normalized = str(pro_account_text or args.lucidflex_account).strip().lower().replace(",", "").replace("$", "")
    try:
        args.lucidpro_account = int(float(pro_normalized[:-1]) * 1000) if pro_normalized.endswith("k") else int(pro_normalized)
    except ValueError:
        p.error("LucidPro account size must be 25k, 50k, 100k, or 150k")
    if args.lucidpro_account not in {25000, 50000, 100000, 150000}:
        p.error("LucidPro account size must be 25k, 50k, 100k, or 150k")
    if args.lucidpro_dll is None and any(asset != "BTC" for asset in assets):
        dll_default = "on" if bool(cfg.get("lucidpro", {}).get("daily_loss_limit", False)) else "off"
        dll_text = _ask("LucidPro optional Daily Loss Limit? (on/off)", dll_default).strip().lower()
        if dll_text not in {"on", "off"}:
            p.error("LucidPro Daily Loss Limit must be 'on' or 'off'")
        args.lucidpro_dll = dll_text == "on"
    else:
        args.lucidpro_dll = str(args.lucidpro_dll or "off").lower() == "on"
    print(f"Strategy: {args.strategy.name} | timeframe: {args.bar_timeframe} | data: {args.data_period} | assets: {', '.join(assets)} | NumContracts: {args.num_contracts:g} | LucidFlex: {args.lucidflex_account:,} | LucidPro: {args.lucidpro_account:,} (DLL {'on' if args.lucidpro_dll else 'off'})", flush=True)
    for asset in assets:
        _run_asset(args, p, cfg, asset, strategy_source)

if __name__ == "__main__":
    main()
