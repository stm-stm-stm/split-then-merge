"""Index OpenVid-1M zip parts *without downloading them*.

OpenVid's ``OpenVid_part<N>.zip`` archives (30-47 GB each) are grouped by
source dataset, so most parts contain no Panda-70M-derived videos at all.
The zip central directory sits at the end of each archive; two HTTP range
requests per part are enough to list its members and rank parts by how many
of our filtered videos they hold.
"""

from __future__ import annotations

import json
import struct
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional

import requests

from .openvid import PART_URL

_EOCD64_LOC_SIG = b"PK\x06\x07"
_EOCD_SIG = b"PK\x05\x06"


def _fetch(url: str, start: int, end: Optional[int] = None, session: Optional[requests.Session] = None) -> bytes:
    s = session or requests
    rng = f"bytes={start}-" + ("" if end is None else str(end))
    r = s.get(url, headers={"Range": rng}, allow_redirects=True, timeout=120)
    r.raise_for_status()
    return r.content


def _content_length(url: str, session: Optional[requests.Session] = None) -> int:
    s = session or requests
    r = s.head(url, allow_redirects=True, timeout=60)
    r.raise_for_status()
    return int(r.headers["Content-Length"])


def list_zip_members_remote(url: str, session: Optional[requests.Session] = None) -> List[str]:
    """Member basenames of a remote zip via range requests (zip64-aware)."""
    size = _content_length(url, session)
    tail_len = min(size, 1 << 16)
    tail = _fetch(url, size - tail_len, size - 1, session)
    eocd = tail.rfind(_EOCD_SIG)
    if eocd < 0:
        raise ValueError("EOCD not found")
    cd_size, cd_offset = struct.unpack("<II", tail[eocd + 12 : eocd + 20])
    loc = tail.rfind(_EOCD64_LOC_SIG, 0, eocd)
    if cd_offset == 0xFFFFFFFF or cd_size == 0xFFFFFFFF or loc >= 0:
        (eocd64_offset,) = struct.unpack("<Q", tail[loc + 8 : loc + 16])
        if eocd64_offset >= size - tail_len:
            rec = tail[eocd64_offset - (size - tail_len) :]
        else:
            rec = _fetch(url, eocd64_offset, eocd64_offset + 64, session)
        cd_size, cd_offset = struct.unpack("<QQ", rec[40:56])
    cd = _fetch(url, cd_offset, cd_offset + cd_size - 1, session)
    return [e["name"] for e in parse_central_directory(cd)]


def parse_central_directory(cd: bytes) -> List[Dict]:
    """Entries of a central directory blob: name (basename), header_offset, csize, usize, method (zip64-aware)."""
    entries: List[Dict] = []
    pos = 0
    while pos + 46 <= len(cd) and cd[pos : pos + 4] == b"PK\x01\x02":
        (method,) = struct.unpack("<H", cd[pos + 10 : pos + 12])
        csize, usize = struct.unpack("<II", cd[pos + 20 : pos + 28])
        n_name, n_extra, n_comment = struct.unpack("<HHH", cd[pos + 28 : pos + 34])
        (header_offset,) = struct.unpack("<I", cd[pos + 42 : pos + 46])
        name = cd[pos + 46 : pos + 46 + n_name].decode("utf-8", "replace")
        extra = cd[pos + 46 + n_name : pos + 46 + n_name + n_extra]
        # zip64 extended information (id 0x0001): values replace the 0xFFFFFFFF placeholders in order
        e = 0
        while e + 4 <= len(extra):
            eid, elen = struct.unpack("<HH", extra[e : e + 4])
            if eid == 0x0001:
                body, b = extra[e + 4 : e + 4 + elen], 0
                if usize == 0xFFFFFFFF:
                    (usize,) = struct.unpack("<Q", body[b : b + 8])
                    b += 8
                if csize == 0xFFFFFFFF:
                    (csize,) = struct.unpack("<Q", body[b : b + 8])
                    b += 8
                if header_offset == 0xFFFFFFFF:
                    (header_offset,) = struct.unpack("<Q", body[b : b + 8])
                    b += 8
            e += 4 + elen
        if not name.endswith("/"):
            entries.append(
                {"name": name.rsplit("/", 1)[-1], "header_offset": header_offset, "csize": csize, "usize": usize, "method": method}
            )
        pos += 46 + n_name + n_extra + n_comment
    return entries


def remote_central_directory(url: str, session: Optional[requests.Session] = None) -> List[Dict]:
    size = _content_length(url, session)
    tail_len = min(size, 1 << 16)
    tail = _fetch(url, size - tail_len, size - 1, session)
    eocd = tail.rfind(_EOCD_SIG)
    cd_size, cd_offset = struct.unpack("<II", tail[eocd + 12 : eocd + 20])
    loc = tail.rfind(_EOCD64_LOC_SIG, 0, eocd)
    if cd_offset == 0xFFFFFFFF or cd_size == 0xFFFFFFFF or loc >= 0:
        (eocd64_offset,) = struct.unpack("<Q", tail[loc + 8 : loc + 16])
        rec = (
            tail[eocd64_offset - (size - tail_len) :]
            if eocd64_offset >= size - tail_len
            else _fetch(url, eocd64_offset, eocd64_offset + 64, session)
        )
        cd_size, cd_offset = struct.unpack("<QQ", rec[40:56])
    return parse_central_directory(_fetch(url, cd_offset, cd_offset + cd_size - 1, session))


def fetch_member(url: str, entry: Dict, dst: Path, session: Optional[requests.Session] = None) -> Path:
    """Download one zip member (local header + data) with range requests; inflate if needed."""
    import zlib

    s = session or requests
    ho = entry["header_offset"]
    head = _fetch(url, ho, ho + 29, s)
    assert head[:4] == b"PK\x03\x04", f"bad local header for {entry['name']}"
    n_name, n_extra = struct.unpack("<HH", head[26:30])
    data_start = ho + 30 + n_name + n_extra
    tmp = dst.with_suffix(dst.suffix + ".part")
    with s.get(url, headers={"Range": f"bytes={data_start}-{data_start + entry['csize'] - 1}"}, stream=True, timeout=300) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            if entry["method"] == 0:
                for chunk in r.iter_content(1 << 20):
                    f.write(chunk)
            elif entry["method"] == 8:
                d = zlib.decompressobj(-15)
                for chunk in r.iter_content(1 << 20):
                    f.write(d.decompress(chunk))
                f.write(d.flush())
            else:
                raise ValueError(f"unsupported compression method {entry['method']}")
    if tmp.stat().st_size != entry["usize"]:
        tmp.unlink(missing_ok=True)
        raise IOError(f"size mismatch for {entry['name']}")
    tmp.replace(dst)
    return dst


def fetch_selected_members(part: int, wanted: Dict[str, str], out_dir: Path, workers: int = 8) -> int:
    """Fetch only the wanted members of ``OpenVid_part<part>.zip`` straight from Hugging Face."""
    url = PART_URL.format(part=part)
    session = requests.Session()
    entries = [e for e in remote_central_directory(url, session) if e["name"] in wanted]
    out_dir.mkdir(parents=True, exist_ok=True)
    todo = [e for e in entries if not (out_dir / e["name"]).exists()]
    print(
        f"[openvid-remote] part{part}: {len(entries)} selected members, {len(todo)} to fetch "
        f"({sum(e['csize'] for e in todo) / 1e9:.2f} GB)",
        flush=True,
    )

    def one(e: Dict):
        dst = out_dir / e["name"]
        for attempt in range(3):
            try:
                fetch_member(url, e, dst, session)
                dst.with_suffix(".json").write_text(json.dumps({"caption": wanted[e["name"]], "origin": f"OpenVid-1M/part{part}"}))
                return 1
            except Exception as exc:  # noqa: BLE001
                if attempt == 2:
                    print(f"[openvid-remote] failed {e['name']}: {exc!r}", flush=True)
        return 0

    n = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for i, r in enumerate(ex.map(one, todo)):
            n += r
            if (i + 1) % 100 == 0:
                print(f"[openvid-remote] part{part}: {i + 1}/{len(todo)} fetched", flush=True)
    print(f"[openvid-remote] part{part}: fetched {n} videos -> {out_dir}", flush=True)
    return n


def index_parts(parts: List[int], cache_path: Path, workers: int = 8) -> Dict[int, List[str]]:
    """Return {part: [member basenames]} using a JSON cache."""
    cache: Dict[int, List[str]] = {}
    if cache_path.exists():
        cache = {int(k): v for k, v in json.loads(cache_path.read_text()).items()}
    todo = [p for p in parts if p not in cache]
    session = requests.Session()

    def one(p: int):
        try:
            return p, list_zip_members_remote(PART_URL.format(part=p), session)
        except Exception as exc:  # part may not exist (split parts, HTTP errors)
            print(f"[openvid-index] part{p}: {exc!r}")
            return p, None

    with ThreadPoolExecutor(max_workers=workers) as ex:
        for p, names in ex.map(one, todo):
            if names is not None:
                cache[p] = names
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(cache))
    return cache


def rank_parts(index: Dict[int, List[str]], wanted: set) -> List[Dict]:
    rows = []
    for p, names in index.items():
        hit = sum(1 for n in names if n in wanted)
        rows.append({"part": p, "members": len(names), "selected": hit, "yield": hit / max(len(names), 1)})
    rows.sort(key=lambda r: -r["selected"])
    return rows
