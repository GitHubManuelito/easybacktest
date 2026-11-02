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
    source_upper = source.upper()
    hist_slots: set[int] = set()
    depth_slots: set[int] = set()
    gex_slots: set[int] = set()
    cot_names = set(re.findall(r"['\"]([A-Z0-9_]+_(?:TFF|LEG)_[A-Z0-9_]+)['\"]", source_upper))
    if "OFB_GETAT" in source_upper:
        hist_slots.update(int(x) for x in re.findall(r"OFB_GETAT\s*\(\s*(\d+)", source_upper))
    if "OFB_POC(" in source_upper:
        hist_slots.update(POC_SLOTS.values())
    if "OFB_VAH(" in source_upper:
        hist_slots.update(VAH_SLOTS.values())
    if "OFB_VAL(" in source_upper:
        hist_slots.update(VAL_SLOTS.values())
    if "OFB_PRIORPOC(" in source_upper:
        hist_slots.update(PRIOR_POC_SLOTS.values())
    if "OFB_VWAP_SD(" in source_upper:
        hist_slots.update(VWAP_SD_SLOTS.values())
    if "OFB_VWAP(" in source_upper:
        hist_slots.update(VWAP_SLOTS.values())
    if "OFB_BIDDEPTH(" in source_upper:
        hist_slots.update((3, 48, 50, 52))
    if "OFB_ASKDEPTH(" in source_upper:
        hist_slots.update((4, 49, 51, 53))
    if "OFB_BIDWALLPX(" in source_upper:
        hist_slots.update((54, 56))
    if "OFB_BIDWALLSZ(" in source_upper:
        hist_slots.update((55, 57))
    if "OFB_ASKWALLPX(" in source_upper:
        hist_slots.update((58, 60))
    if "OFB_ASKWALLSZ(" in source_upper:
        hist_slots.update((59, 61))
    if "OFB_BIGTRADE(" in source_upper:
        hist_slots.add(46)
    if "OFB_BIGTRADEPX(" in source_upper:
        hist_slots.add(47)
    for match in re.finditer(r"OFB_GEXSCALAR\s*\(\s*(\d+)", source_upper):
        gex_slots.add(int(match.group(1)))

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
