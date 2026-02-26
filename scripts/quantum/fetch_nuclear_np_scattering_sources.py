from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import tarfile
from dataclasses import dataclass
from pathlib import Path
from urllib.request import Request, urlopen


@dataclass(frozen=True)
class FileSpec:
    url: str
    relpath: str
    expected_bytes: int | None = None


def _sha256(path: Path, *, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk_bytes)
            # 条件分岐: `not b` を満たす経路を評価する。
            if not b:
                break

            h.update(b)

    return h.hexdigest()


def _download(url: str, out_path: Path, *, expected_bytes: int | None, max_bytes: int) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # 条件分岐: `out_path.exists() and out_path.stat().st_size > 0` を満たす経路を評価する。
    if out_path.exists() and out_path.stat().st_size > 0:
        # 条件分岐: `expected_bytes is None or out_path.stat().st_size == expected_bytes` を満たす経路を評価する。
        if expected_bytes is None or out_path.stat().st_size == expected_bytes:
            print(f"[skip] exists: {out_path}")
            return

        print(f"[redo] size mismatch: {out_path} ({out_path.stat().st_size} != {expected_bytes})")
        out_path.unlink()

    req = Request(url, headers={"User-Agent": "waveP/quantum-fetch"})
    with urlopen(req, timeout=30) as resp:
        total = resp.headers.get("Content-Length")
        total_i = int(total) if total is not None else None
        # 条件分岐: `total_i is not None and total_i > max_bytes` を満たす経路を評価する。
        if total_i is not None and total_i > max_bytes:
            raise SystemExit(
                f"[fail] refusing to download large file ({total_i} bytes > max={max_bytes} bytes):\n"
                f"  {url}\n"
                "Re-run with a larger --max-gib if you really want this download."
            )

        with out_path.open("wb") as f:
            f.write(resp.read())

    # 条件分岐: `expected_bytes is not None and out_path.stat().st_size != expected_bytes` を満たす経路を評価する。

    if expected_bytes is not None and out_path.stat().st_size != expected_bytes:
        raise RuntimeError(f"downloaded size mismatch: {out_path} ({out_path.stat().st_size} != {expected_bytes})")

    # 条件分岐: `out_path.stat().st_size == 0` を満たす経路を評価する。

    if out_path.stat().st_size == 0:
        raise RuntimeError(f"downloaded empty file: {out_path}")

    print(f"[ok] downloaded: {out_path} ({out_path.stat().st_size} bytes)")


def _parse_paren_uncertainty(num: str) -> dict[str, float]:
    """
    Parse values like:
      5.424(4)  -> value=5.424, sigma=0.004
      -23.748(10) -> value=-23.748, sigma=0.010
      2.75(5) -> value=2.75, sigma=0.05
    If no "(...)" is present, sigma is omitted.
    """
    m = re.fullmatch(r"\s*([+-]?\d+(?:\.\d+)?)(?:\((\d+)\))?\s*", num)
    # 条件分岐: `not m` を満たす経路を評価する。
    if not m:
        raise ValueError(f"invalid numeric token: {num!r}")

    value_str = m.group(1)
    value = float(value_str)
    paren = m.group(2)
    out: dict[str, float] = {"value": float(value)}
    # 条件分岐: `paren is not None` を満たす経路を評価する。
    if paren is not None:
        decimals = len(value_str.split(".", 1)[1]) if "." in value_str else 0
        out["sigma"] = float(int(paren) * (10 ** (-decimals)))

    return out


def _extract_equation_block(tex: str, *, label: str) -> str:
    label_token = f"\\label{{{label}}}"
    idx_label = tex.find(label_token)
    # 条件分岐: `idx_label < 0` を満たす経路を評価する。
    if idx_label < 0:
        raise ValueError(f"label not found: {label_token}")

    idx_begin = tex.rfind("\\begin{equation}", 0, idx_label)
    # 条件分岐: `idx_begin < 0` を満たす経路を評価する。
    if idx_begin < 0:
        raise ValueError(f"begin{{equation}} not found for label={label}")

    idx_end = tex.find("\\end{equation}", idx_label)
    # 条件分岐: `idx_end < 0` を満たす経路を評価する。
    if idx_end < 0:
        raise ValueError(f"end{{equation}} not found for label={label}")

    idx_end += len("\\end{equation}")
    return tex[idx_begin:idx_end]


def _extract_params_from_block(block: str) -> dict[str, object]:
    def find_num(key: str, pat: str) -> dict[str, float] | None:
        m = re.search(pat, block)
        # 条件分岐: `not m` を満たす経路を評価する。
        if not m:
            return None

        return _parse_paren_uncertainty(m.group(1))

    out: dict[str, object] = {}
    out["a_t_fm"] = find_num("a_t_fm", r"a_\{t\}\s*=\s*([+-]?\d+(?:\.\d+)?(?:\(\d+\))?)")
    out["r_t_fm"] = find_num("r_t_fm", r"r_\{t\}\s*=\s*([+-]?\d+(?:\.\d+)?(?:\(\d+\))?)")
    out["a_s_fm"] = find_num("a_s_fm", r"a_\{s\}\s*=\s*([+-]?\d+(?:\.\d+)?(?:\(\d+\))?)")
    out["r_s_fm"] = find_num("r_s_fm", r"r_\{s\}\s*=\s*([+-]?\d+(?:\.\d+)?(?:\(\d+\))?)")
    out["v2t_fm3"] = find_num("v2t_fm3", r"v_\{2t\}\s*=\s*([+-]?\d+(?:\.\d+)?(?:\(\d+\))?)")
    out["v2s_fm3"] = find_num("v2s_fm3", r"v_\{2s\}\s*=\s*([+-]?\d+(?:\.\d+)?(?:\(\d+\))?)")

    missing = [k for k in ("a_t_fm", "r_t_fm", "a_s_fm", "r_s_fm") if out.get(k) is None]
    # 条件分岐: `missing` を満たす経路を評価する。
    if missing:
        raise ValueError(f"missing required params in block: {missing}")

    return out


def _read_member_from_tar_gz(tar_gz_path: Path, *, member_name: str) -> str:
    with tar_gz_path.open("rb") as f:
        data = f.read()

    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tf:
        try:
            member = tf.getmember(member_name)
        except KeyError as e:
            raise ValueError(f"missing tar member: {member_name}") from e

        extracted = tf.extractfile(member)
        # 条件分岐: `extracted is None` を満たす経路を評価する。
        if extracted is None:
            raise ValueError(f"unable to extract tar member: {member_name}")

        return extracted.read().decode("utf-8", errors="replace")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Fetch and freeze primary sources for Phase 7 / Step 7.9.2 (np scattering low-energy parameters). "
            "Downloads arXiv PDF+source for 0704.1024v1 and extracts eq.(16–19) parameters into JSON."
        )
    )
    ap.add_argument("--offline", action="store_true", help="Do not download; only verify expected files exist.")
    ap.add_argument(
        "--max-gib",
        type=float,
        default=0.25,
        help="Refuse single-file downloads larger than this (GiB). Default: 0.25",
    )
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[2]
    src_dir = root / "data" / "quantum" / "sources"
    src_dir.mkdir(parents=True, exist_ok=True)

    arxiv_id = "0704.1024v1"
    pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    src_url = f"https://arxiv.org/e-print/{arxiv_id}"

    files: list[FileSpec] = [
        FileSpec(url=pdf_url, relpath=f"arxiv_{arxiv_id}.pdf"),
        FileSpec(url=src_url, relpath=f"arxiv_{arxiv_id}_src.tar.gz"),
    ]

    max_bytes = int(float(args.max_gib) * (1024**3))
    # 条件分岐: `not args.offline` を満たす経路を評価する。
    if not args.offline:
        for spec in files:
            _download(spec.url, src_dir / spec.relpath, expected_bytes=spec.expected_bytes, max_bytes=max_bytes)

    missing: list[Path] = []
    for spec in files:
        p = src_dir / spec.relpath
        # 条件分岐: `not p.exists() or p.stat().st_size == 0` を満たす経路を評価する。
        if not p.exists() or p.stat().st_size == 0:
            missing.append(p)

    # 条件分岐: `missing` を満たす経路を評価する。

    if missing:
        raise SystemExit("[fail] missing files:\n" + "\n".join(f"- {p}" for p in missing))

    # Extract low-energy parameters from arXiv source (article.tex).

    src_tar = src_dir / f"arxiv_{arxiv_id}_src.tar.gz"
    tex = _read_member_from_tar_gz(src_tar, member_name="article.tex")
    sets: list[dict[str, object]] = []
    for label, kind in [
        ("16", "legacy_compilation_set_1"),
        ("17", "legacy_compilation_set_2"),
        ("18", "gwu_said_phase_shifts"),
        ("19", "nijmegen_phase_shifts"),
    ]:
        block = _extract_equation_block(tex, label=label)
        params = _extract_params_from_block(block)
        sets.append({"eq_label": int(label), "kind": kind, "params": params})

    extracted = {
        "generated_utc": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
        "dataset": "np scattering low-energy parameters (effective-range expansion) from arXiv:0704.1024v1",
        "arxiv_id": arxiv_id,
        "doi": "10.1134/S1063778807040072",
        "extraction": {
            "source": {"tar_member": "article.tex", "path": str(src_tar)},
            "equations_extracted": [16, 17, 18, 19],
            "notes": [
                "Eq.(16–17): two parameter sets quoted as 'experimental' in older literature (with parentheses uncertainties).",
                "Eq.(18–19): parameters computed from GWU/SAID vs Nijmegen phase shifts (no uncertainties in the source).",
                "This JSON is derived from the arXiv source tarball for reproducible offline parsing.",
            ],
        },
        "parameter_sets": sets,
    }

    out_extracted = src_dir / "np_scattering_low_energy_arxiv_0704_1024v1_extracted.json"
    out_extracted.write_text(json.dumps(extracted, ensure_ascii=False, indent=2), encoding="utf-8")

    manifest = {
        "generated_utc": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
        "dataset": "Phase 7 / Step 7.9.2 primary sources (np scattering low-energy parameters)",
        "notes": [
            "Primary source is arXiv:0704.1024v1 (PDF + source tarball).",
            "The extracted JSON is derived from the tarball and is used by analysis scripts for offline reproducibility.",
        ],
        "files": [],
    }

    def add_file(*, url: str | None, path: Path, extra: dict[str, object] | None = None) -> None:
        item = {"url": url, "path": str(path), "bytes": int(path.stat().st_size), "sha256": _sha256(path)}
        # 条件分岐: `extra` を満たす経路を評価する。
        if extra:
            item.update(extra)

        manifest["files"].append(item)

    for spec in files:
        add_file(url=spec.url, path=src_dir / spec.relpath)

    add_file(url=None, path=out_extracted, extra={"derived_from": str(src_tar), "tar_member": "article.tex"})

    out_manifest = src_dir / "np_scattering_low_energy_arxiv_0704_1024v1_manifest.json"
    out_manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[ok] extracted: {out_extracted}")
    print(f"[ok] manifest : {out_manifest}")


# 条件分岐: `__name__ == "__main__"` を満たす経路を評価する。

if __name__ == "__main__":
    main()
