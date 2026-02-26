from __future__ import annotations

import argparse
import contextlib
import json
import math
import struct
import zipfile
import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np


_REC_DTYPE = np.dtype([("ch", "u1"), ("t", "<u8"), ("sec", "<u2")], align=False)  # 11 bytes/rec


@dataclass(frozen=True)
class Config:
    # NIST timetag bin: 78.125 ps (12.8 GHz clock)
    seconds_per_timetag: float = 78.125e-12

    # Coincidence-window sweep (ns); used for a simple time-difference pairing study.
    windows_ns: tuple[float, ...] = (
        0.5,
        1.0,
        2.0,
        5.0,
        10.0,
        20.0,
        50.0,
        100.0,
        200.0,
        500.0,
        1000.0,
        2000.0,
        5000.0,
    )

    # Stop early for dev/debug (None = read full file).
    max_seconds: int | None = None


def _ks_distance(x: np.ndarray, y: np.ndarray) -> float:
    # Two-sample KS statistic without scipy.
    if x.size == 0 or y.size == 0:
        return float("nan")

    xs = np.sort(x)
    ys = np.sort(y)
    n = xs.size
    m = ys.size
    i = 0
    j = 0
    d = 0.0
    while i < n and j < m:
        # 条件分岐: `xs[i] <= ys[j]` を満たす経路を評価する。
        if xs[i] <= ys[j]:
            i += 1
        else:
            j += 1

        d = max(d, abs(i / n - j / m))

    d = max(d, abs(1.0 - j / m), abs(i / n - 1.0))
    return float(d)


@dataclass(frozen=True)
class SideEvents:
    click_t: np.ndarray  # timetag counts
    click_setting: np.ndarray  # 0/1
    click_delay: np.ndarray  # timetag counts from last sync
    pps_t: np.ndarray  # timetag counts
    counts_by_channel: dict[int, int]


@dataclass(frozen=True)
class _ZipCdEntry:
    filename: str
    flag: int
    method: int
    disk_start: int
    local_header_offset: int
    compressed_size: int
    uncompressed_size: int


class _MultiFileStream:
    def __init__(
        self, *, parts: list[Path], start_disk: int, start_offset: int, max_bytes: int, chunk_bytes: int = 8 * 1024 * 1024
    ) -> None:
        # 条件分岐: `not parts` を満たす経路を評価する。
        if not parts:
            raise ValueError("parts must not be empty")

        # 条件分岐: `start_disk < 0 or start_disk >= len(parts)` を満たす経路を評価する。

        if start_disk < 0 or start_disk >= len(parts):
            raise ValueError(f"invalid start_disk={start_disk} for parts={len(parts)}")

        self._parts = parts
        self._disk = start_disk
        self._off = start_offset
        self._remaining = int(max_bytes)
        self._chunk_bytes = int(chunk_bytes)
        self._f = self._parts[self._disk].open("rb")
        self._f.seek(self._off)

    def read(self, n: int = -1) -> bytes:
        # 条件分岐: `self._remaining <= 0` を満たす経路を評価する。
        if self._remaining <= 0:
            return b""

        # 条件分岐: `n is None or n < 0` を満たす経路を評価する。

        if n is None or n < 0:
            n = min(self._remaining, self._chunk_bytes)

        n = min(int(n), self._remaining)
        # 条件分岐: `n <= 0` を満たす経路を評価する。
        if n <= 0:
            return b""

        out = bytearray()
        need = n
        while need > 0 and self._remaining > 0:
            part = self._parts[self._disk]
            part_size = int(part.stat().st_size)
            cur = int(self._f.tell())
            avail = part_size - cur
            # 条件分岐: `avail <= 0` を満たす経路を評価する。
            if avail <= 0:
                self._advance_disk()
                continue

            take = min(need, avail, self._remaining)
            b = self._f.read(take)
            # 条件分岐: `not b` を満たす経路を評価する。
            if not b:
                self._advance_disk()
                continue

            out += b
            need -= len(b)
            self._remaining -= len(b)

        return bytes(out)

    def _advance_disk(self) -> None:
        self._f.close()
        self._disk += 1
        # 条件分岐: `self._disk >= len(self._parts)` を満たす経路を評価する。
        if self._disk >= len(self._parts):
            # No more parts.
            self._remaining = 0
            return

        self._f = self._parts[self._disk].open("rb")
        self._f.seek(0)

    def close(self) -> None:
        try:
            self._f.close()
        except Exception:
            pass


class _DeflateReader:
    def __init__(self, raw: _MultiFileStream) -> None:
        self._raw = raw
        self._z = zlib.decompressobj(-15)  # raw DEFLATE stream (zip format)
        self._buf = bytearray()
        self._eof = False

    def read(self, n: int = -1) -> bytes:
        # 条件分岐: `n == 0` を満たす経路を評価する。
        if n == 0:
            return b""

        # 条件分岐: `n is not None and n > 0 and len(self._buf) >= n` を満たす経路を評価する。

        if n is not None and n > 0 and len(self._buf) >= n:
            out = bytes(self._buf[:n])
            del self._buf[:n]
            return out

        # Fill buffer until we have enough (or reach EOF).

        while not self._eof and (n is None or n < 0 or len(self._buf) < n):
            chunk = self._raw.read(1024 * 1024)
            # 条件分岐: `not chunk` を満たす経路を評価する。
            if not chunk:
                self._buf += self._z.flush()
                self._eof = True
                break

            self._buf += self._z.decompress(chunk)
            # 条件分岐: `self._z.eof` を満たす経路を評価する。
            if self._z.eof:
                self._buf += self._z.flush()
                self._eof = True
                break

        # 条件分岐: `n is None or n < 0` を満たす経路を評価する。

        if n is None or n < 0:
            out = bytes(self._buf)
            self._buf.clear()
            return out

        out = bytes(self._buf[:n])
        del self._buf[:n]
        return out

    def close(self) -> None:
        try:
            self._raw.close()
        except Exception:
            pass


def _parse_eocd_tail(tail: bytes) -> tuple[int, int, int, int, int, int, int]:
    idx = tail.rfind(b"PK\x05\x06")
    # 条件分岐: `idx < 0` を満たす経路を評価する。
    if idx < 0:
        raise ValueError("EOCD not found")

    sig, disk, disk_cd, n_this, n_total, cd_size, cd_offset, comment_len = struct.unpack("<4sHHHHIIH", tail[idx : idx + 22])
    # 条件分岐: `sig != b"PK\x05\x06"` を満たす経路を評価する。
    if sig != b"PK\x05\x06":
        raise ValueError("bad EOCD signature")

    return idx, disk, disk_cd, n_total, cd_size, cd_offset, comment_len


def _parse_zip64_locator(tail: bytes) -> tuple[int, int, int, int] | None:
    idx = tail.rfind(b"PK\x06\x07")
    # 条件分岐: `idx < 0` を満たす経路を評価する。
    if idx < 0:
        return None

    sig, disk_start, rec_offset, n_disks = struct.unpack("<4sIQI", tail[idx : idx + 20])
    # 条件分岐: `sig != b"PK\x06\x07"` を満たす経路を評価する。
    if sig != b"PK\x06\x07":
        return None

    return idx, disk_start, int(rec_offset), int(n_disks)


def _read_zip64_eocd(path: Path, *, disk_start: int, rec_offset: int) -> dict[str, int]:
    # For our use we only need cd_size/cd_offset/n_total, but parse the full header for sanity.
    with path.open("rb") as f:
        f.seek(rec_offset)
        b = f.read(56)

    # 条件分岐: `len(b) < 56 or b[:4] != b"PK\x06\x06"` を満たす経路を評価する。

    if len(b) < 56 or b[:4] != b"PK\x06\x06":
        raise ValueError("zip64 EOCD record not found at locator offset")

    sig, sz, ver_made, ver_need, disk, disk_cd, n_this, n_total, cd_size, cd_offset = struct.unpack("<4sQHHIIQQQQ", b[:56])
    # 条件分岐: `sig != b"PK\x06\x06"` を満たす経路を評価する。
    if sig != b"PK\x06\x06":
        raise ValueError("bad zip64 EOCD signature")

    return {
        "disk": int(disk),
        "disk_cd": int(disk_cd),
        "n_total": int(n_total),
        "cd_size": int(cd_size),
        "cd_offset": int(cd_offset),
        "disk_start": int(disk_start),
        "rec_offset": int(rec_offset),
        "size": int(sz),
        "ver_made": int(ver_made),
        "ver_need": int(ver_need),
        "n_this": int(n_this),
    }


def _parse_cd_entry(cd: bytes, *, pos: int) -> tuple[_ZipCdEntry, int]:
    # 条件分岐: `cd[pos : pos + 4] != b"PK\x01\x02"` を満たす経路を評価する。
    if cd[pos : pos + 4] != b"PK\x01\x02":
        raise ValueError("central directory entry signature not found")

    (
        _sig,
        _ver_made,
        _ver_need,
        flag,
        method,
        _mtime,
        _mdate,
        _crc,
        csz32,
        usz32,
        name_len,
        extra_len,
        comment_len,
        disk_start16,
        _int_attr,
        _ext_attr,
        lh_offset32,
    ) = struct.unpack("<4sHHHHHHIIIHHHHHII", cd[pos : pos + 46])

    name_start = pos + 46
    name_end = name_start + int(name_len)
    extra_end = name_end + int(extra_len)
    comment_end = extra_end + int(comment_len)
    # 条件分岐: `comment_end > len(cd)` を満たす経路を評価する。
    if comment_end > len(cd):
        raise ValueError("central directory entry truncated")

    name = cd[name_start:name_end].decode("utf-8", errors="replace")
    extra = cd[name_end:extra_end]

    disk_start = int(disk_start16)
    lh_offset = int(lh_offset32)
    csz = int(csz32)
    usz = int(usz32)

    # Zip64 extended information extra field (0x0001).
    if any(v in (0xFFFFFFFF,) for v in (csz32, usz32, lh_offset32)) or disk_start16 == 0xFFFF:
        p = 0
        while p + 4 <= len(extra):
            eid, elen = struct.unpack("<HH", extra[p : p + 4])
            data = extra[p + 4 : p + 4 + elen]
            p += 4 + elen
            # 条件分岐: `eid != 0x0001` を満たす経路を評価する。
            if eid != 0x0001:
                continue

            # 条件分岐: `usz32 == 0xFFFFFFFF and len(data) >= 8` を満たす経路を評価する。

            if usz32 == 0xFFFFFFFF and len(data) >= 8:
                usz = int(struct.unpack("<Q", data[:8])[0])
                data = data[8:]

            # 条件分岐: `csz32 == 0xFFFFFFFF and len(data) >= 8` を満たす経路を評価する。

            if csz32 == 0xFFFFFFFF and len(data) >= 8:
                csz = int(struct.unpack("<Q", data[:8])[0])
                data = data[8:]

            # 条件分岐: `lh_offset32 == 0xFFFFFFFF and len(data) >= 8` を満たす経路を評価する。

            if lh_offset32 == 0xFFFFFFFF and len(data) >= 8:
                lh_offset = int(struct.unpack("<Q", data[:8])[0])
                data = data[8:]

            # 条件分岐: `disk_start16 == 0xFFFF and len(data) >= 4` を満たす経路を評価する。

            if disk_start16 == 0xFFFF and len(data) >= 4:
                disk_start = int(struct.unpack("<I", data[:4])[0])

            break

    return (
        _ZipCdEntry(
            filename=name,
            flag=int(flag),
            method=int(method),
            disk_start=disk_start,
            local_header_offset=lh_offset,
            compressed_size=csz,
            uncompressed_size=usz,
        ),
        comment_end,
    )


def _open_single_member_stream_multipart(zip_last: Path) -> tuple[object, str]:
    # Find sibling parts: "<...>.z01, .z02, ... , .zip". We expect the caller to pass the final ".zip".
    prefix = str(zip_last)
    # 条件分岐: `not prefix.lower().endswith(".zip")` を満たす経路を評価する。
    if not prefix.lower().endswith(".zip"):
        raise ValueError(f"expected .zip path, got: {zip_last}")

    prefix = prefix[:-4]  # drop ".zip"

    parts: list[Path] = []
    for i in range(1, 100):
        p = Path(f"{prefix}.z{i:02d}")
        # 条件分岐: `p.exists()` を満たす経路を評価する。
        if p.exists():
            parts.append(p)
            continue
        # Stop at the first gap once we have at least one part.

        if parts:
            break

    # 条件分岐: `not parts` を満たす経路を評価する。

    if not parts:
        raise FileNotFoundError(f"multipart .z01 not found for: {zip_last}")

    disks = parts + [zip_last]

    # Parse EOCD from the last disk to locate the central directory.
    with zip_last.open("rb") as f:
        f.seek(0, 2)
        size = int(f.tell())
        tail_bytes = min(size, 128 * 1024)
        f.seek(-tail_bytes, 2)
        tail = f.read(tail_bytes)

    _, disk, disk_cd, n_total, cd_size, cd_offset, _comment_len = _parse_eocd_tail(tail)
    zip64 = _parse_zip64_locator(tail)
    # 条件分岐: `zip64 is not None and (n_total == 0xFFFF or cd_size == 0xFFFFFFFF or cd_offse...` を満たす経路を評価する。
    if zip64 is not None and (n_total == 0xFFFF or cd_size == 0xFFFFFFFF or cd_offset == 0xFFFFFFFF):
        _, disk_start, rec_offset, n_disks = zip64
        # 条件分岐: `disk_start != disk_cd` を満たす経路を評価する。
        if disk_start != disk_cd:
            raise ValueError("zip64 locator on unexpected disk (unsupported)")

        zip64_rec = _read_zip64_eocd(zip_last, disk_start=disk_start, rec_offset=rec_offset)
        disk = zip64_rec["disk"]
        disk_cd = zip64_rec["disk_cd"]
        n_total = zip64_rec["n_total"]
        cd_size = zip64_rec["cd_size"]
        cd_offset = zip64_rec["cd_offset"]

    # 条件分岐: `disk + 1 != len(disks)` を満たす経路を評価する。

    if disk + 1 != len(disks):
        raise ValueError(f"multipart disk count mismatch: EOCD says {disk+1}, found {len(disks)} files")

    # 条件分岐: `disk_cd != len(disks) - 1` を満たす経路を評価する。

    if disk_cd != len(disks) - 1:
        raise ValueError("central directory not on final .zip disk (unsupported)")

    # Read central directory bytes from the final disk.

    with zip_last.open("rb") as f:
        f.seek(int(cd_offset))
        cd = f.read(int(cd_size))

    entries: list[_ZipCdEntry] = []
    pos = 0
    for _ in range(int(n_total)):
        ent, pos = _parse_cd_entry(cd, pos=pos)
        entries.append(ent)

    # 条件分岐: `len(entries) != 1` を満たす経路を評価する。

    if len(entries) != 1:
        raise ValueError(f"expected exactly 1 file in multipart zip, got {len(entries)}: {zip_last}")

    ent = entries[0]

    # 条件分岐: `ent.disk_start < 0 or ent.disk_start >= len(disks)` を満たす経路を評価する。
    if ent.disk_start < 0 or ent.disk_start >= len(disks):
        raise ValueError(f"invalid disk_start={ent.disk_start} for disks={len(disks)}")

    # Read local header to find the data start offset.

    with disks[ent.disk_start].open("rb") as f:
        f.seek(int(ent.local_header_offset))
        hdr = f.read(30)

    # 条件分岐: `len(hdr) < 30 or hdr[:4] != b"PK\x03\x04"` を満たす経路を評価する。

    if len(hdr) < 30 or hdr[:4] != b"PK\x03\x04":
        raise ValueError("local header not found in multipart zip")

    (
        _sig,
        _ver_need,
        _flag,
        _method,
        _mtime,
        _mdate,
        _crc,
        _csz32,
        _usz32,
        name_len,
        extra_len,
    ) = struct.unpack("<4sHHHHHIIIHH", hdr[:30])
    data_start = int(ent.local_header_offset) + 30 + int(name_len) + int(extra_len)

    raw = _MultiFileStream(
        parts=disks,
        start_disk=int(ent.disk_start),
        start_offset=int(data_start),
        max_bytes=int(ent.compressed_size),
    )
    # 条件分岐: `ent.method == 0` を満たす経路を評価する。
    if ent.method == 0:
        return raw, ent.filename

    # 条件分岐: `ent.method == 8` を満たす経路を評価する。

    if ent.method == 8:
        return _DeflateReader(raw), ent.filename

    raw.close()
    raise ValueError(f"unsupported compression method in multipart zip: {ent.method}")


@contextlib.contextmanager
def _open_single_member_stream(zip_path: Path):  # noqa: ANN001
    # Normal single-file zip
    try:
        with zipfile.ZipFile(zip_path, "r") as z:
            names = z.namelist()
            # 条件分岐: `len(names) != 1` を満たす経路を評価する。
            if len(names) != 1:
                raise RuntimeError(f"expected 1 entry in zip, got {len(names)}: {zip_path}")

            name = names[0]
            with z.open(name, "r") as f:
                yield f, name

            return
    except zipfile.BadZipFile:
        pass

    # Multipart zip (".z01/.z02... + .zip").

    stream, name = _open_single_member_stream_multipart(zip_path)
    try:
        yield stream, name
    finally:
        close = getattr(stream, "close", None)
        # 条件分岐: `callable(close)` を満たす経路を評価する。
        if callable(close):
            close()


def _read_side(zip_path: Path, *, max_seconds: int | None) -> SideEvents:
    # 条件分岐: `not zip_path.exists()` を満たす経路を評価する。
    if not zip_path.exists():
        raise FileNotFoundError(zip_path)

    counts: dict[int, int] = {}
    click_t_chunks: list[np.ndarray] = []
    click_setting_chunks: list[np.ndarray] = []
    click_delay_chunks: list[np.ndarray] = []
    pps_t_chunks: list[np.ndarray] = []

    last_sync: int = -1
    last_setting: int = -1
    stop = False

    with _open_single_member_stream(zip_path) as (f, _name):
        leftover = b""
        while True:
            b = f.read(16 * 1024 * 1024)
            # 条件分岐: `not b` を満たす経路を評価する。
            if not b:
                break

            data = leftover + b
            n = len(data) // _REC_DTYPE.itemsize
            # 条件分岐: `n == 0` を満たす経路を評価する。
            if n == 0:
                leftover = data
                continue

            use = memoryview(data)[: n * _REC_DTYPE.itemsize]
            leftover = data[n * _REC_DTYPE.itemsize :]

            rec = np.frombuffer(use, dtype=_REC_DTYPE)
            ch = rec["ch"]
            t = rec["t"].astype(np.int64, copy=False)
            sec = rec["sec"]

            # counts by channel
            u, c = np.unique(ch, return_counts=True)
            for ui, ci in zip(u.tolist(), c.tolist(), strict=True):
                counts[int(ui)] = counts.get(int(ui), 0) + int(ci)

            # optional early stop

            if max_seconds is not None:
                over = np.flatnonzero(sec > max_seconds)
                # 条件分岐: `over.size` を満たす経路を評価する。
                if over.size:
                    cut = int(over[0])
                    ch = ch[:cut]
                    t = t[:cut]
                    stop = True

            # 条件分岐: `ch.size == 0` を満たす経路を評価する。

            if ch.size == 0:
                break

            # Last sync time (channel 6) for each record.

            sync_vals = np.where(ch == 6, t, -1)
            sync_acc = np.maximum.accumulate(sync_vals)
            # 条件分岐: `last_sync >= 0` を満たす経路を評価する。
            if last_sync >= 0:
                sync_acc = np.maximum(sync_acc, last_sync)

            # Last setting (channels 2/4) for each record.

            set_evt = np.full(ch.shape, -1, dtype=np.int8)
            set_evt[ch == 2] = 0
            set_evt[ch == 4] = 1
            set_idx = np.where(set_evt >= 0, np.arange(ch.size, dtype=np.int64), -1)
            last_set_idx = np.maximum.accumulate(set_idx)
            set_filled = np.where(last_set_idx >= 0, set_evt[last_set_idx], -1).astype(np.int8, copy=False)

            # 条件分岐: `last_setting >= 0` を満たす経路を評価する。
            if last_setting >= 0:
                set_filled = np.where(last_set_idx >= 0, set_filled, last_setting).astype(np.int8, copy=False)

            # Click events (channel 0) with sync+setting defined.

            click_mask = ch == 0
            valid = click_mask & (sync_acc >= 0) & (set_filled >= 0)
            # 条件分岐: `np.any(valid)` を満たす経路を評価する。
            if np.any(valid):
                t_click = t[valid]
                delay = t_click - sync_acc[valid]
                ok = delay >= 0
                # 条件分岐: `np.any(ok)` を満たす経路を評価する。
                if np.any(ok):
                    click_t_chunks.append(t_click[ok].astype(np.int64, copy=False))
                    click_delay_chunks.append(delay[ok].astype(np.int64, copy=False))
                    click_setting_chunks.append(set_filled[valid][ok].astype(np.int8, copy=False))

            # PPS events (channel 5)

            pps_mask = ch == 5
            # 条件分岐: `np.any(pps_mask)` を満たす経路を評価する。
            if np.any(pps_mask):
                pps_t_chunks.append(t[pps_mask].astype(np.int64, copy=False))

            # Persist last values for next chunk.

            last_sync = int(sync_acc[-1])
            # 条件分岐: `np.any(set_evt >= 0)` を満たす経路を評価する。
            if np.any(set_evt >= 0):
                last_setting = int(set_evt[np.flatnonzero(set_evt >= 0)[-1]])

            # 条件分岐: `stop` を満たす経路を評価する。

            if stop:
                break

    click_t = np.concatenate(click_t_chunks) if click_t_chunks else np.zeros((0,), dtype=np.int64)
    click_setting = (
        np.concatenate(click_setting_chunks) if click_setting_chunks else np.zeros((0,), dtype=np.int8)
    )
    click_delay = np.concatenate(click_delay_chunks) if click_delay_chunks else np.zeros((0,), dtype=np.int64)
    pps_t = np.concatenate(pps_t_chunks) if pps_t_chunks else np.zeros((0,), dtype=np.int64)

    return SideEvents(
        click_t=click_t,
        click_setting=click_setting,
        click_delay=click_delay,
        pps_t=pps_t,
        counts_by_channel={int(k): int(v) for k, v in sorted(counts.items())},
    )


def _estimate_pps_offset(pps_a: np.ndarray, pps_b: np.ndarray, *, max_shift: int = 5) -> dict[str, float | int]:
    # 条件分岐: `pps_a.size < 20 or pps_b.size < 20` を満たす経路を評価する。
    if pps_a.size < 20 or pps_b.size < 20:
        raise RuntimeError(
            "not enough PPS events to estimate offset "
            f"(need >=20; got a={pps_a.size}, b={pps_b.size}). "
            "Increase --max-seconds (or analyze a longer segment)."
        )

    best = None
    for shift in range(-max_shift, max_shift + 1):
        a0 = max(0, shift)
        b0 = max(0, -shift)
        n = min(pps_a.size - a0, pps_b.size - b0)
        # 条件分岐: `n < 20` を満たす経路を評価する。
        if n < 20:
            continue

        diffs = pps_a[a0 : a0 + n] - pps_b[b0 : b0 + n]
        med = float(np.median(diffs))
        mad = float(np.median(np.abs(diffs - med)))
        cand = (mad, abs(shift), med, shift, n)
        # 条件分岐: `best is None or cand < best` を満たす経路を評価する。
        if best is None or cand < best:
            best = cand

    # 条件分岐: `best is None` を満たす経路を評価する。

    if best is None:
        raise RuntimeError("failed to align PPS sequences")

    mad, _, med, shift, n = best
    return {"offset_counts": int(round(med)), "shift": int(shift), "mad_counts": mad, "n_used": int(n)}


def _coincidence_sweep(
    a_t: np.ndarray,
    a_set: np.ndarray,
    b_t: np.ndarray,
    b_set: np.ndarray,
    windows_counts: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    # Greedy pairing on sorted event streams.
    # Returns counts[w, a_setting, b_setting] and total pairs per window.
    order_a = np.argsort(a_t, kind="mergesort")
    order_b = np.argsort(b_t, kind="mergesort")
    at = a_t[order_a]
    aset = a_set[order_a]
    bt = b_t[order_b]
    bset = b_set[order_b]

    counts = np.zeros((windows_counts.size, 2, 2), dtype=np.int64)
    pairs = np.zeros((windows_counts.size,), dtype=np.int64)
    for wi, w in enumerate(windows_counts.astype(np.int64).tolist()):
        i = 0
        j = 0
        c = np.zeros((2, 2), dtype=np.int64)
        p = 0
        while i < at.size and j < bt.size:
            dt = bt[j] - at[i]
            # 条件分岐: `dt < -w` を満たす経路を評価する。
            if dt < -w:
                j += 1
                continue

            # 条件分岐: `dt > w` を満たす経路を評価する。

            if dt > w:
                i += 1
                continue

            c[int(aset[i]), int(bset[j])] += 1
            p += 1
            i += 1
            j += 1

        counts[wi] = c
        pairs[wi] = p

    return counts, pairs


def main() -> None:
    parser = argparse.ArgumentParser(description="Reanalyze NIST Bell test time-tag data (coincidence/window bias).")
    parser.add_argument(
        "--alice-zip",
        type=Path,
        default=None,
        help="Path to Alice *.dat.compressed.zip (default: cached training set).",
    )
    parser.add_argument(
        "--bob-zip",
        type=Path,
        default=None,
        help="Path to Bob *.dat.compressed.zip (default: cached training set).",
    )
    parser.add_argument("--max-seconds", type=int, default=None, help="Stop after this many transfer-seconds.")
    parser.add_argument(
        "--out-tag",
        default="",
        help=(
            "Optional suffix for output filenames to avoid overwriting "
            '(e.g. "run_03_43"). Default: empty (= historical fixed filenames).'
        ),
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    src_dir = root / "data" / "quantum" / "sources" / "nist_belltestdata"
    out_dir = root / "output" / "public" / "quantum"
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = Config(max_seconds=args.max_seconds)

    def _tag(name: str, *, ext: str) -> Path:
        stem = name if not args.out_tag else f"{name}__{args.out_tag}"
        return out_dir / f"{stem}.{ext}"

    alice_zip = args.alice_zip or (
        src_dir
        / "compressed"
        / "alice"
        / "2015_09_18"
        / "03_31_CH_pockel_100kHz.run4.afterTimingfix2_training.alice.dat.compressed.zip"
    )
    bob_zip = args.bob_zip or (
        src_dir
        / "compressed"
        / "bob"
        / "2015_09_18"
        / "03_31_CH_pockel_100kHz.run4.afterTimingfix2_training.bob.dat.compressed.zip"
    )

    print(f"[info] alice: {alice_zip}")
    print(f"[info] bob  : {bob_zip}")

    a = _read_side(alice_zip, max_seconds=cfg.max_seconds)
    b = _read_side(bob_zip, max_seconds=cfg.max_seconds)

    align = _estimate_pps_offset(a.pps_t, b.pps_t, max_shift=5)
    offset_counts = int(align["offset_counts"])
    offset_seconds = offset_counts * cfg.seconds_per_timetag

    b_click_t_aligned = b.click_t + offset_counts
    b_pps_t_aligned = b.pps_t + offset_counts

    # Delay distributions (ns) by local setting.
    a_delay_ns = a.click_delay.astype(np.float64) * cfg.seconds_per_timetag * 1e9
    b_delay_ns = b.click_delay.astype(np.float64) * cfg.seconds_per_timetag * 1e9

    a0 = a_delay_ns[a.click_setting == 0]
    a1 = a_delay_ns[a.click_setting == 1]
    b0 = b_delay_ns[b.click_setting == 0]
    b1 = b_delay_ns[b.click_setting == 1]

    ks_a = _ks_distance(a0, a1)
    ks_b = _ks_distance(b0, b1)

    # Coincidence sweep on aligned click timestamps (simple greedy pairing).
    windows_counts = np.asarray(
        [w * 1e-9 / cfg.seconds_per_timetag for w in cfg.windows_ns], dtype=np.float64
    )
    counts, pairs = _coincidence_sweep(a.click_t, a.click_setting, b_click_t_aligned, b.click_setting, windows_counts)

    # Output CSV (window vs counts).
    out_csv = _tag("nist_belltest_coincidence_sweep", ext="csv")
    header = [
        "window_ns",
        "pairs_total",
        "c00",
        "c01",
        "c10",
        "c11",
    ]
    lines = [",".join(header)]
    for wi, w_ns in enumerate(cfg.windows_ns):
        c = counts[wi]
        row = [
            f"{float(w_ns):g}",
            str(int(pairs[wi])),
            str(int(c[0, 0])),
            str(int(c[0, 1])),
            str(int(c[1, 0])),
            str(int(c[1, 1])),
        ]
        lines.append(",".join(row))

    out_csv.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Plot
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(13.5, 4.2), dpi=150)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 1.35])

    def _delay_panel(ax, title: str, d0: np.ndarray, d1: np.ndarray) -> None:
        # Focus on the bulk region to show setting-dependent shifts.
        allv = np.concatenate([d0, d1]) if d0.size and d1.size else (d0 if d0.size else d1)
        # 条件分岐: `allv.size == 0` を満たす経路を評価する。
        if allv.size == 0:
            ax.set_title(title + " (no data)")
            return

        hi = float(np.percentile(allv, 99.5))
        hi = max(50.0, min(hi, 2000.0))
        bins = np.linspace(0.0, hi, 120)
        ax.hist(d0, bins=bins, alpha=0.55, label="setting=0", color="#1f77b4")
        ax.hist(d1, bins=bins, alpha=0.55, label="setting=1", color="#ff7f0e")
        ax.set_title(title)
        ax.set_xlabel("click delay from sync (ns)")
        ax.set_ylabel("count")
        ax.grid(True, ls=":", lw=0.6, alpha=0.6)
        ax.legend(frameon=True, fontsize=9)

    ax0 = fig.add_subplot(gs[0, 0])
    _delay_panel(ax0, "Alice: click delay vs setting", a0, a1)

    ax1 = fig.add_subplot(gs[0, 1])
    _delay_panel(ax1, "Bob: click delay vs setting", b0, b1)

    ax2 = fig.add_subplot(gs[0, 2])
    w = np.asarray(cfg.windows_ns, dtype=float)
    ax2.plot(w, pairs, marker="o", lw=1.5, label="pairs total")
    ax2.plot(w, counts[:, 0, 0], marker="o", lw=1.2, label="c00")
    ax2.plot(w, counts[:, 0, 1], marker="o", lw=1.2, label="c01")
    ax2.plot(w, counts[:, 1, 0], marker="o", lw=1.2, label="c10")
    ax2.plot(w, counts[:, 1, 1], marker="o", lw=1.2, label="c11")
    ax2.set_xscale("log")
    ax2.set_xlabel("coincidence window (ns)")
    ax2.set_ylabel("greedy-paired coincidences")
    ax2.set_title("Window dependence (aligned by GPS PPS)")
    ax2.grid(True, which="both", ls=":", lw=0.6, alpha=0.6)
    ax2.legend(frameon=True, fontsize=9, ncol=2)

    fig.suptitle(
        "NIST Bell test (time-tag): setting-dependent delays can bias coincidence selection",
        y=1.02,
        fontsize=12,
    )

    note = f"offset(bob→alice)≈{offset_seconds:.3f} s ({offset_seconds/3600.0:.2f} h); KS(A)={ks_a:.3f}, KS(B)={ks_b:.3f}"
    fig.text(0.01, -0.02, note, fontsize=9)
    fig.tight_layout()

    out_png = _tag("nist_belltest_time_tag_bias", ext="png")
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)

    def _infer_run_base(path: Path, *, side: str) -> str:
        suf = f".{side}.dat.compressed.zip"
        return path.name[: -len(suf)] if path.name.endswith(suf) else path.name

    metrics = {
        "generated_utc": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
        "dataset": {
            "source": "NIST belltestdata (Shalm et al. 2015; time-tag repository)",
            "run_base_alice": _infer_run_base(alice_zip, side="alice"),
            "run_base_bob": _infer_run_base(bob_zip, side="bob"),
            "alice_zip": str(alice_zip),
            "bob_zip": str(bob_zip),
        },
        "config": {
            "seconds_per_timetag": cfg.seconds_per_timetag,
            "windows_ns": list(map(float, cfg.windows_ns)),
            "pairing": "greedy (1-1 pairing on sorted click streams)",
            "max_seconds": cfg.max_seconds,
            "out_tag": str(args.out_tag),
        },
        "alignment": {
            "pps_a_count": int(a.pps_t.size),
            "pps_b_count": int(b.pps_t.size),
            "pps_b_count_aligned": int(b_pps_t_aligned.size),
            **{k: (int(v) if isinstance(v, (int, np.integer)) else float(v)) for k, v in align.items()},
            "offset_seconds": float(offset_seconds),
        },
        "counts": {
            "alice_click_total": int(a.click_t.size),
            "bob_click_total": int(b.click_t.size),
            "alice_click_by_setting": {
                "0": int(np.count_nonzero(a.click_setting == 0)),
                "1": int(np.count_nonzero(a.click_setting == 1)),
            },
            "bob_click_by_setting": {
                "0": int(np.count_nonzero(b.click_setting == 0)),
                "1": int(np.count_nonzero(b.click_setting == 1)),
            },
            "alice_counts_by_channel": a.counts_by_channel,
            "bob_counts_by_channel": b.counts_by_channel,
        },
        "delay_stats_ns": {
            "alice": {
                "ks_setting0_vs_1": ks_a,
                "setting0_median": float(np.median(a0)) if a0.size else float("nan"),
                "setting1_median": float(np.median(a1)) if a1.size else float("nan"),
            },
            "bob": {
                "ks_setting0_vs_1": ks_b,
                "setting0_median": float(np.median(b0)) if b0.size else float("nan"),
                "setting1_median": float(np.median(b1)) if b1.size else float("nan"),
            },
        },
        "coincidence_sweep": {
            "csv": str(out_csv),
            "windows_ns": list(map(float, cfg.windows_ns)),
            "pairs_total": list(map(int, pairs.tolist())),
            "counts": {
                "c00": list(map(int, counts[:, 0, 0].tolist())),
                "c01": list(map(int, counts[:, 0, 1].tolist())),
                "c10": list(map(int, counts[:, 1, 0].tolist())),
                "c11": list(map(int, counts[:, 1, 1].tolist())),
            },
        },
        "outputs": {"png": str(out_png), "csv": str(out_csv)},
        "notes": [
            "This script does NOT reproduce the official loophole-free p-value calculation.",
            "It demonstrates that detection-time distributions depend on settings, which can make coincidence selection setting-dependent.",
            "Pairing algorithm choice matters; here we use a simple greedy 1-1 match on click timestamps.",
        ],
    }
    out_json = _tag("nist_belltest_time_tag_bias_metrics", ext="json")
    out_json.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[ok] png : {out_png}")
    print(f"[ok] csv : {out_csv}")
    print(f"[ok] json: {out_json}")


# 条件分岐: `__name__ == "__main__"` を満たす経路を評価する。

if __name__ == "__main__":
    main()
