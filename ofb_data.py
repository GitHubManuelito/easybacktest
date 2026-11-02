"""Readers for the local OrderFlowBridge (OFB) market-data bundle.

The bridge stores little-endian binary records. This module validates those
records and attaches only requested feature slots to the OHLCV frame, so a
multi-million-row depth feed is never expanded into 128 pandas columns.
"""
from __future__ import annotations

import json
import mmap
import re
import struct
from pathlib import Path

import numpy as np
import pandas as pd


FILE_LAYOUTS = {
    "ofb_hist.bin": (b"3HBF", 80),
    "ofb_depth.bin": (b"3DBF", 128),
    # GEX rows currently carry 273 float64 payload values. Header slot 128 is
    # the strike capacity, not the number of float values in each row.
    "ofb_gex.bin": (b"1GBF", None),
}

POC_SLOTS = {1: 24, 2: 18, 3: 20, 4: 22, 5: 26, 6: 66, 7: 70}
VAH_SLOTS = {1: 28, 2: 34, 3: 64, 4: 30, 5: 32, 6: 68, 7: 72}
VAL_SLOTS = {1: 29, 2: 35, 3: 65, 4: 31, 5: 33, 6: 69, 7: 73}
PRIOR_POC_SLOTS = {1: 25, 2: 19, 3: 21, 4: 23, 5: 27, 6: 67, 7: 71}
VWAP_SLOTS = {1: 42, 2: 36, 3: 38, 4: 40, 5: 44}
VWAP_SD_SLOTS = {1: 43, 2: 37, 3: 39, 4: 41, 5: 45}

GEX_SCALAR_SLOTS = {
    "centroid": 0, "major_positive": 1, "major_negative": 2,
    "centroid_day": 3, "positive_day": 4, "negative_day": 5,
    "positive_oi": 6, "negative_oi": 7, "basis": 8, "dte": 9,
    "flip": 10, "flip_day": 11, "flip_oi": 12, "spot": 13,
    "flip_steep": 14, "flip_steep_day": 15, "flip_steep_oi": 16,
}


class OFBDataProvider:
    """Lazy native reader for the documented OrderFlowBridge data APIs.

    The depth file is several GB, so it stays memory-mapped. Depth records store
    64 levels as four packed float32 values per level (bid price/size, ask
    price/size); GEX stores 17 float64 scalars followed by 128 packed
    {strike, gamma, call OI, put OI} float32 records.
    """

    def __init__(self, folder: Path, timezone: str = "America/New_York"):
        self.folder = Path(folder)
        self.timezone = timezone
        self._tables = {}
        self._cot = None
        self._cot_names = None

    def _open_table(self, name: str, magic: bytes, fields: int | None = None):
        if name in self._tables:
            return self._tables[name]
        path = self.folder / name
        if not path.is_file():
            return None
        handle, mapping, rows, actual_fields, metadata = _checked_record(path, magic, fields)
        stride = 8 + actual_fields * 8
        keys = np.ndarray((rows,), dtype="<u8", buffer=mapping, offset=16, strides=(stride,))
        table = {"handle": handle, "mapping": mapping, "rows": rows,
                 "fields": actual_fields, "metadata": metadata, "stride": stride, "keys": keys}
        self._tables[name] = table
        return table

    @staticmethod
    def _row(table, key):
        if table is None:
            return None
        if key is None or int(float(key)) <= 0:
            return table["rows"] - 1
        key_value = int(float(key))
        keys = table["keys"]
        index = int(np.searchsorted(keys, key_value, side="left"))
        return index if index < len(keys) and int(keys[index]) == key_value else None

    @staticmethod
    def _value(table, slot: int, row: int):
        if table is None or row is None or not 0 <= int(slot) < table["fields"]:
            return float("nan")
        offset = 16 + row * table["stride"] + 8 + int(slot) * 8
        return float(struct.unpack_from("<d", table["mapping"], offset)[0])

    def hist_count(self):
        table = self._open_table("ofb_hist.bin", b"3HBF", 80)
        return 0 if table is None else table["rows"]

    def gex_count(self):
        table = self._open_table("ofb_gex.bin", b"1GBF")
        return 0 if table is None else table["rows"]

    def depth_count(self):
        table = self._open_table("ofb_depth.bin", b"3DBF", 128)
        return 0 if table is None else table["rows"]

    def cot_data(self):
        if self._cot is not None:
            return self._cot
        path = self.folder / "ofb_cot.bin"
        if not path.is_file():
            return None
        with path.open("rb") as f:
            header = f.read(16)
            if len(header) != 16 or header[:4] != b"1CBF":
                raise ValueError("ofb_cot.bin: invalid header")
            _, rows, count = struct.unpack_from("<III", header, 4)
            raw_names = f.read(count * 32)
            names = [raw_names[i * 32:(i + 1) * 32].split(b"\0", 1)[0].decode("ascii", "replace").upper()
                     for i in range(count)]
            row_bytes = 8 + count * 8
            if path.stat().st_size != 16 + count * 32 + rows * row_bytes:
                raise ValueError("ofb_cot.bin: row width does not match series metadata")
            payload = f.read()
        keys = np.ndarray((rows,), dtype="<u8", buffer=payload, offset=0, strides=(row_bytes,)).copy()
        values = np.ndarray((rows, count), dtype="<f8", buffer=payload, offset=8,
                            strides=(row_bytes, 8)).copy()
        self._cot_names = {name: i for i, name in enumerate(names)}
        self._cot = (keys, values)
        return self._cot

    def cot_index(self, name):
        self.cot_data()
        return self._cot_names.get(str(name).strip().upper(), -1) if self._cot_names else -1

    def cot_at(self, index, key):
        data = self.cot_data()
        if data is None or not 0 <= int(index) < data[1].shape[1]:
            return 0.0
        keys, values = data
        row = int(np.searchsorted(keys, int(float(key)), side="right")) - 1
        return float(values[row, int(index)]) if row >= 0 else 0.0

    def cot_has(self, index, key):
        data = self.cot_data()
        if data is None or not 0 <= int(index) < data[1].shape[1]:
            return 0
        keys = data[0]
        return int(len(keys) > 0 and int(key) >= int(keys[0]))

    def cot_percentile(self, name, weeks, key):
        index = self.cot_index(name)
        data = self.cot_data()
        weeks = int(float(weeks))
        if index < 0 or data is None or weeks < 1:
            return 50.0, 0, 0.0
        keys, values = data
        key = int(key)
        base = int(np.searchsorted(keys, key, side="right")) - 1
        try:
            key_time = pd.to_datetime(f"{key:014d}", format="%Y%m%d%H%M%S")
        except (TypeError, ValueError):
            return 50.0, 0, 0.0
        week_key = int((key_time - pd.Timedelta(days=weeks * 7)).strftime("%Y%m%d%H%M%S"))
        far = int(np.searchsorted(keys, week_key, side="right")) - 1
        if base < 0:
            return 50.0, 0, 0.0
        net = float(values[base, index])
        if far < 0:
            return 50.0, 0, net
        # The Bridge helper samples one as-of observation at each weekly offset.
        sample = []
        for offset in range(1, weeks + 1):
            probe = int((key_time - pd.Timedelta(days=offset * 7)).strftime("%Y%m%d%H%M%S"))
            row = int(np.searchsorted(keys, probe, side="right")) - 1
            if row < 0:
                return 50.0, 0, net
            sample.append(float(values[row, index]))
        return 100.0 * sum(value < net for value in sample) / weeks, 1, net

    def hist_at(self, slot, key=None):
        table = self._open_table("ofb_hist.bin", b"3HBF", 80)
        return self._value(table, slot, self._row(table, key))

    def gex_scalar(self, slot, key=None):
        table = self._open_table("ofb_gex.bin", b"1GBF")
        row = self._row(table, key)
        return self._value(table, slot, row)

    def gex_at(self, rank, what, key=None):
        table = self._open_table("ofb_gex.bin", b"1GBF")
        row = self._row(table, key)
        rank, what = int(rank), int(what)
        if table is None or row is None or not 0 <= rank < min(128, int(table["metadata"])) or not 0 <= what <= 3:
            return float("nan")
        offset = 16 + row * table["stride"] + 8 + 17 * 8 + rank * 16 + what * 4
        return float(struct.unpack_from("<f", table["mapping"], offset)[0])

    def depth_at(self, side, level, what, key=None):
        table = self._open_table("ofb_depth.bin", b"3DBF", 128)
        row = self._row(table, key)
        side, level, what = int(side), int(level), int(what)
        if table is None or row is None or side not in (0, 1) or what not in (0, 1) or not 0 <= level < 64:
            return float("nan")
        offset = 16 + row * table["stride"] + 8 + level * 16 + side * 8 + what * 4
        return float(struct.unpack_from("<f", table["mapping"], offset)[0])

    def close(self):
        for table in self._tables.values():
            table["keys"] = None
            table["mapping"].close()
            table["handle"].close()
        self._tables.clear()


def _checked_record(path: Path, expected_magic: bytes, expected_fields: int | None = None):
    """Return a read-only mmap and record dimensions after strict validation."""
    handle = path.open("rb")
    header = handle.read(16)
    if len(header) != 16 or header[:4] != expected_magic:
        handle.close()
        raise ValueError(f"{path.name}: unexpected OFB header or truncated file")
    version, rows, metadata = struct.unpack_from("<III", header, 4)
    if version not in (1, 5) or rows <= 0:
        handle.close()
        raise ValueError(f"{path.name}: invalid OFB version/row count")
    record_bytes = (path.stat().st_size - 16) // rows
    if record_bytes < 16 or (path.stat().st_size - 16) % rows:
        handle.close()
        raise ValueError(f"{path.name}: record data length does not match its header")
    fields = (record_bytes - 8) // 8
    if 8 + fields * 8 != record_bytes or (expected_fields is not None and fields != expected_fields):
        handle.close()
        raise ValueError(f"{path.name}: unsupported record width ({fields} float fields)")
    mapping = mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ)
    return handle, mapping, rows, fields, metadata


def _key_to_utc(keys: np.ndarray, timezone: str) -> np.ndarray:
    # Bridge key format is YYYYMMDDHHMMSS, in the MultiCharts chart timezone.
    text = keys.astype("uint64", copy=False).astype(str)
    parsed = pd.to_datetime(text, format="%Y%m%d%H%M%S", errors="coerce")
    if pd.isna(parsed).any():
        raise ValueError("OFB feed contains an invalid timestamp key")
    local = pd.DatetimeIndex(parsed).tz_localize(timezone, ambiguous="infer", nonexistent="shift_forward")
    return local.tz_convert("UTC").asi8


def inspect_ofb_bundle(folder: Path) -> dict:
    """Read bundle metadata without loading the 9 GB payloads into memory."""
    report = {"folder": str(folder), "files": {}}
    for name, (magic, expected) in FILE_LAYOUTS.items():
        path = folder / name
        if not path.is_file():
            continue
        with path.open("rb") as f:
            header = f.read(16)
            if len(header) != 16 or header[:4] != magic:
                report["files"][name] = {"valid": False, "error": "header/magic mismatch"}
                continue
        handle, mapping, rows, fields, metadata = _checked_record(path, magic, expected)
        try:
            stride = 8 + fields * 8
            first = struct.unpack_from("<Q", mapping, 16)[0]
            last = struct.unpack_from("<Q", mapping, 16 + (rows - 1) * stride)[0]
        finally:
            mapping.close()
            handle.close()
        report["files"][name] = {"valid": True, "version": struct.unpack_from("<I", header, 4)[0],
                                  "rows": rows, "values_per_row": fields, "metadata": metadata,
                                  "bytes": path.stat().st_size, "first_key": first, "last_key": last}
    cot = folder / "ofb_cot.bin"
    if cot.is_file():
        with cot.open("rb") as f:
            header = f.read(16)
            if len(header) == 16 and header[:4] == b"1CBF":
                version, rows, series_count = struct.unpack_from("<III", header, 4)
                name_bytes = f.read(series_count * 32)
                names = [name_bytes[i * 32:(i + 1) * 32].split(b"\0", 1)[0].decode("ascii", "replace")
                         for i in range(series_count)]
                row_bytes = 8 + series_count * 8
                valid = (cot.stat().st_size == 16 + series_count * 32 + rows * row_bytes)
                report["files"][cot.name] = {"valid": valid, "version": version, "rows": rows,
                                             "series_count": series_count, "series_names": names,
                                             "bytes": cot.stat().st_size}
    no_days = folder / "ofb_no_data_days.json"
    if no_days.is_file():
        report["coverage_exceptions"] = json.loads(no_days.read_text(encoding="utf-8"))
    return report


def _attach_slot_file(frame: pd.DataFrame, path: Path, slots: set[int], timezone: str,
                      prefix: str, magic: bytes, fields: int) -> pd.DataFrame:
    if not slots:
        return frame
    if any(slot < 0 or slot >= fields for slot in slots):
        raise ValueError(f"requested {prefix} slot outside file range 0..{fields - 1}")
    handle, mapping, rows, _, _ = _checked_record(path, magic, fields)
    keys = values = None
    try:
        stride = 8 + fields * 8
        keys = np.ndarray((rows,), dtype="<u8", buffer=mapping, offset=16, strides=(stride,))
        record_times = _key_to_utc(keys, timezone)
        bar_times = pd.DatetimeIndex(frame.timestamp).asi8
        indices = np.searchsorted(record_times, bar_times)
        valid = indices < rows
        valid_positions = np.flatnonzero(valid)
        valid[valid_positions] &= record_times[indices[valid_positions]] == bar_times[valid_positions]
        row_ids = indices[valid]
        data = {}
        for slot in sorted(slots):
            values = np.ndarray((rows,), dtype="<f8", buffer=mapping,
                                offset=24 + slot * 8, strides=(stride,))
            out = np.full(len(frame), np.nan, dtype=np.float64)
            out[valid] = values[row_ids]
            data[f"{prefix}_{slot}"] = out
        result = pd.concat([frame.reset_index(drop=True), pd.DataFrame(data)], axis=1)
        del keys, values
        return result
    finally:
        keys = values = None
        mapping.close()
        handle.close()


def _attach_cot(frame: pd.DataFrame, path: Path, series_names: set[str], timezone: str) -> pd.DataFrame:
    if not series_names:
        return frame
    with path.open("rb") as f:
        header = f.read(16)
        if len(header) != 16 or header[:4] != b"1CBF":
            raise ValueError("ofb_cot.bin: invalid header")
        _, rows, series_count = struct.unpack_from("<III", header, 4)
        name_block = f.read(series_count * 32)
        all_names = [name_block[i * 32:(i + 1) * 32].split(b"\0", 1)[0].decode("ascii", "replace")
                     for i in range(series_count)]
        selected = {name.upper() for name in series_names}
        indices = [all_names.index(name) for name in selected if name in all_names]
        if not indices:
            return frame
        stride = 8 + series_count * 8
        if path.stat().st_size != 16 + series_count * 32 + rows * stride:
            raise ValueError("ofb_cot.bin: row width does not match series metadata")
        mapping = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        keys = values = None
        try:
            offset = 16 + series_count * 32
            keys = np.ndarray((rows,), dtype="<u8", buffer=mapping, offset=offset,
                              strides=(stride,))
            feed_times = _key_to_utc(keys, timezone)
            bars = pd.DatetimeIndex(frame.timestamp).asi8
            # COT is an as-of release feed. Never expose a report before its timestamp.
            positions = np.searchsorted(feed_times, bars, side="right") - 1
            valid = positions >= 0
            result = frame.reset_index(drop=True).copy()
            for column in indices:
                values = np.ndarray((rows,), dtype="<f8", buffer=mapping,
                                    offset=offset + 8 + column * 8, strides=(stride,))
                out = np.full(len(result), np.nan, dtype=np.float64)
                out[valid] = values[positions[valid]]
                result[f"ofb_cot_{all_names[column].lower()}"] = out
            del keys, values
            return result
        finally:
            keys = values = None
            mapping.close()


def attach_ofb_features(frame: pd.DataFrame, folder: Path, source: str,
                        timezone: str = "America/New_York") -> pd.DataFrame:
    """Attach documented OFB slots referenced by a PowerLanguage script.

    Direct ``OFB_GetAt(slot, ...)`` references load only that slot. Helper names
    from the Bridge examples load their documented slot groups. Unselected raw
    depth/GEX ladder values remain available in their original binary files.
    """
    analysis_source = re.sub(r"\{.*?\}", " ", source, flags=re.S)
    analysis_source = re.sub(r"//[^\n]*", " ", analysis_source)
    source_upper = analysis_source.upper()
    hist_slots: set[int] = set()
    depth_slots: set[int] = set()
    gex_slots: set[int] = set()
    cot_names = set(re.findall(r"['\"]([A-Z0-9_]+_(?:TFF|LEG)_[A-Z0-9_]+)['\"]", source_upper))
    if "OFB_GETAT" in source_upper:
        hist_slots.update(int(x) for x in re.findall(r"OFB_GETAT\s*\(\s*(\d+)", source_upper))
    input_block = re.search(r"(?is)\binputs?\s*:(.*?)(?=\b(?:vars?|variables?|arrays?)\s*:|$)", analysis_source)
    input_defaults = ({name.lower(): float(value) for name, value in re.findall(
        r"\b([A-Za-z_]\w*)\s*\(\s*(-?\d+(?:\.\d+)?)\s*\)", input_block.group(1)
    )} if input_block else {})

    def selected_modes(helper, supported, default):
        calls = list(re.finditer(rf"\b{helper}\s*\(([^()]*)\)", source_upper))
        if not calls:
            return set()
        selected = set()
        for call in calls:
            arg = call.group(1).strip()
            if not arg:
                mode = default
            elif re.fullmatch(r"\d+", arg):
                mode = int(arg)
            elif arg.lower() in input_defaults:
                mode = int(input_defaults[arg.lower()])
            else:
                return set(supported)
            if mode in supported:
                selected.add(mode)
        return selected

    for helper, mapping, default in (
        ("OFB_POC", POC_SLOTS, 1), ("OFB_VAH", VAH_SLOTS, 1),
        ("OFB_VAL", VAL_SLOTS, 1), ("OFB_PRIORPOC", PRIOR_POC_SLOTS, 1),
        ("OFB_VWAP_SD", VWAP_SD_SLOTS, 1), ("OFB_VWAP", VWAP_SLOTS, 1),
    ):
        hist_slots.update(mapping[mode] for mode in selected_modes(helper, mapping, default))

    def selected_numeric_argument(helper, default, supported, parse):
        calls = list(re.finditer(rf"\b{helper}\s*\(([^()]*)\)", source_upper))
        if not calls:
            return set()
        selected = set()
        for call in calls:
            arg = call.group(1).strip()
            if not arg:
                value = default
            elif re.fullmatch(r"\d+(?:\.\d+)?", arg):
                value = float(arg)
            elif arg.lower() in input_defaults:
                value = input_defaults[arg.lower()]
            else:
                return set(supported)
            selected.add(parse(value))
        return selected

    depth_modes = {"OFB_BIDDEPTH": {5: 3, 10: 48, 20: 50, 50: 52},
                   "OFB_ASKDEPTH": {5: 4, 10: 49, 20: 51, 50: 53}}
    for helper, slots in depth_modes.items():
        chosen = selected_numeric_argument(helper, 5, (5, 10, 20, 50),
                                           lambda value: max((level for level in (5, 10, 20, 50) if value >= level), default=5))
        hist_slots.update(slots[level] for level in chosen)
    for helper, slots in (("OFB_BIDWALLPX", {1: 54, 2: 56}), ("OFB_BIDWALLSZ", {1: 55, 2: 57}),
                          ("OFB_ASKWALLPX", {1: 58, 2: 60}), ("OFB_ASKWALLSZ", {1: 59, 2: 61})):
        ranks = selected_numeric_argument(helper, 1, (1, 2), lambda value: min(2, max(1, int(value))))
        hist_slots.update(slots[rank] for rank in ranks)
    if "OFB_BIGTRADE(" in source_upper:
        hist_slots.add(46)
    if "OFB_BIGTRADEPX(" in source_upper:
        hist_slots.add(47)
    for match in re.finditer(r"OFB_GEXSCALAR\s*\(\s*(\d+)", source_upper):
        gex_slots.add(int(match.group(1)))
    for name, slot in {
        "OFB_GEXCENTROID": 0, "OFB_GEXMAJPOS": 1, "OFB_GEXMAJNEG": 2,
        "OFB_GEXCENTROIDDAY": 3, "OFB_GEXPOSDAY": 4, "OFB_GEXNEGDAY": 5,
        "OFB_GEXPOSOI": 6, "OFB_GEXNEGOI": 7, "OFB_GEXBASIS": 8,
        "OFB_GEXDTE": 9, "OFB_GEXFLIP": 10, "OFB_GEXFLIPDAY": 11,
        "OFB_GEXFLIPOI": 12, "OFB_GEXSPOT": 13, "OFB_GEXFLIPSTEEP": 14,
        "OFB_GEXFLIPSTEEPDAY": 15, "OFB_GEXFLIPSTEEPOI": 16,
    }.items():
        if re.search(rf"\b{name}\b", source_upper):
            gex_slots.add(slot)

    result = frame
    hist = folder / "ofb_hist.bin"
    if hist_slots and hist.is_file():
        result = _attach_slot_file(result, hist, hist_slots, timezone, "ofb_hist",
                                   b"3HBF", 80)
    gex = folder / "ofb_gex.bin"
    if gex_slots and gex.is_file():
        with gex.open("rb") as f:
            header = f.read(16)
        rows = struct.unpack_from("<I", header, 8)[0]
        fields = (gex.stat().st_size - 16) // rows
        if fields % 8:
            raise ValueError("ofb_gex.bin: unsupported GEX row width")
        result = _attach_slot_file(result, gex, gex_slots, timezone, "ofb_gex",
                                   b"1GBF", fields // 8 - 1)
    cot = folder / "ofb_cot.bin"
    if cot_names and cot.is_file():
        result = _attach_cot(result, cot, cot_names, timezone)
    return result
