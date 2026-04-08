#!/usr/bin/env python3
"""Windows-friendly path and artifact naming helpers.

Purpose:
- Keep newly generated artifact filenames inside conservative Windows limits.
- Provide response-file expansion for CLI tools that would otherwise hit
  long command-line strings on PowerShell / Win32.
- Offer one shared policy so future scripts do not reintroduce long stems.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Callable, Iterable, Sequence


ROOT = Path(__file__).resolve().parents[2]


WINDOWS_SAFE_PATH_LIMIT = 220
WINDOWS_SAFE_NAME_LIMIT = 120
WINDOWS_SAFE_STEM_LIMIT = 72
WINDOWS_SAFE_APPLY_PATCH_FILES_PER_BATCH = 1
WINDOWS_SAFE_APPLY_PATCH_HEADER_LIMIT = 4096

_NON_ALNUM = re.compile(r"[^0-9A-Za-z]+")
_MULTI_UNDERSCORE = re.compile(r"_+")


# 関数: `slugify_fragment` の入出力契約と処理意図を定義する。
def slugify_fragment(text: str) -> str:
    """Convert free text into a compact ASCII slug fragment."""
    normalized = _NON_ALNUM.sub("_", text.strip().lower())
    normalized = _MULTI_UNDERSCORE.sub("_", normalized).strip("_")
    return normalized or "artifact"


# 関数: `compact_slug` の入出力契約と処理意図を定義する。

def compact_slug(text: str, max_length: int) -> str:
    """Return a stable slug truncated with a digest when needed."""
    slug = slugify_fragment(text)
    # 条件分岐: `len(slug) <= max_length` を満たす経路を評価する。
    if len(slug) <= max_length:
        return slug

    digest = hashlib.sha1(slug.encode("utf-8")).hexdigest()[:10]
    head_limit = max(8, int(max_length) - len(digest) - 1)
    head = slug[:head_limit].rstrip("_")
    return f"{head}_{digest}"


# 関数: `step_slug` の入出力契約と処理意図を定義する。

def step_slug(step_tag: str) -> str:
    """Convert one roadmap step tag into a compact slug."""
    return compact_slug(step_tag.replace(".", "_").replace("-", "_"), 24)


# 関数: `build_compact_artifact_stem` の入出力契約と処理意図を定義する。

def build_compact_artifact_stem(
    step_tag: str,
    artifact_label: str,
    *,
    prefix: str = "q",
    max_stem_length: int = WINDOWS_SAFE_STEM_LIMIT,
) -> str:
    """Build one compact artifact stem from a step tag and a short label."""
    parts = [slugify_fragment(prefix), step_slug(step_tag), slugify_fragment(artifact_label)]
    stem = "_".join(part for part in parts if part)
    return compact_slug(stem, int(max_stem_length))


# 関数: `ensure_windows_path_budget` の入出力契約と処理意図を定義する。

def ensure_windows_path_budget(
    path: Path,
    *,
    max_path_length: int = WINDOWS_SAFE_PATH_LIMIT,
    max_name_length: int = WINDOWS_SAFE_NAME_LIMIT,
) -> Path:
    """Raise when one path exceeds the conservative Windows-safe budget."""
    path_text = str(path)
    # 条件分岐: `len(path.name) > max_name_length` を満たす経路を評価する。
    if len(path.name) > int(max_name_length):
        raise ValueError(
            f"filename exceeds Windows-safe budget: {len(path.name)} > {int(max_name_length)} for {path_text}"
        )

    # 条件分岐: `len(path_text) > max_path_length` を満たす経路を評価する。

    if len(path_text) > int(max_path_length):
        raise ValueError(
            f"path exceeds Windows-safe budget: {len(path_text)} > {int(max_path_length)} for {path_text}"
        )

    return path


# 関数: `build_metrics_paths` の入出力契約と処理意図を定義する。

def build_metrics_paths(
    out_dir: Path,
    stem: str,
    kind: str,
    *,
    max_path_length: int = WINDOWS_SAFE_PATH_LIMIT,
    max_name_length: int = WINDOWS_SAFE_NAME_LIMIT,
) -> dict[str, Path]:
    """Build validated JSON/CSV artifact paths for one metrics payload."""
    json_path = ensure_windows_path_budget(
        out_dir / f"{stem}_{kind}_metrics.json",
        max_path_length=max_path_length,
        max_name_length=max_name_length,
    )
    csv_path = ensure_windows_path_budget(
        out_dir / f"{stem}_{kind}_rows.csv",
        max_path_length=max_path_length,
        max_name_length=max_name_length,
    )
    return {"json": json_path, "csv": csv_path}


# 関数: `normalize_repo_relative_path` の入出力契約と処理意図を定義する。

def normalize_repo_relative_path(path: Path | str, *, repo_root: Path = ROOT) -> str:
    """Return one portable path string preferring repo-relative notation."""
    repo_root_resolved = repo_root.resolve()
    candidate = Path(path)
    anchored = candidate if candidate.is_absolute() else (repo_root_resolved / candidate)
    anchored_resolved = anchored.resolve(strict=False)
    try:
        return anchored_resolved.relative_to(repo_root_resolved).as_posix()
    except ValueError:
        return anchored_resolved.as_posix()


# 関数: `normalize_manifest_entries` の入出力契約と処理意図を定義する。

def normalize_manifest_entries(paths: Sequence[Path | str], *, repo_root: Path = ROOT) -> list[str]:
    """Normalize manifest entries with repo-relative paths when possible."""
    ordered: list[str] = []
    seen: set[str] = set()
    for raw_path in paths:
        normalized = normalize_repo_relative_path(raw_path, repo_root=repo_root)
        if normalized not in seen:
            seen.add(normalized)
            ordered.append(normalized)

    return ordered


# 関数: `write_path_manifest` の入出力契約と処理意図を定義する。

def write_path_manifest(
    path: Path | str,
    entries: Sequence[Path | str],
    *,
    repo_root: Path = ROOT,
    header_lines: Sequence[str] | None = None,
) -> Path:
    """Write one UTF-8 path manifest using compact repo-relative entries."""
    manifest_path = Path(path)
    if not manifest_path.is_absolute():
        manifest_path = (repo_root / manifest_path).resolve()

    ensure_windows_path_budget(manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    output_lines: list[str] = []
    for header_line in header_lines or []:
        stripped = header_line.strip()
        if not stripped:
            continue

        output_lines.append(stripped if stripped.startswith("#") else f"# {stripped}")

    output_lines.extend(normalize_manifest_entries(entries, repo_root=repo_root))
    payload = "\n".join(output_lines)
    if payload:
        payload += "\n"

    manifest_path.write_text(payload, encoding="utf-8", newline="\n")
    return manifest_path


# 関数: `estimate_apply_patch_target_chars` の入出力契約と処理意図を定義する。

def estimate_apply_patch_target_chars(path: Path | str, *, operation: str = "Update") -> int:
    """Estimate header characters added by one apply_patch target."""
    normalized = normalize_repo_relative_path(path)
    return len(f"*** {operation} File: {normalized}\n@@\n")


# 関数: `estimate_apply_patch_header_chars` の入出力契約と処理意図を定義する。

def estimate_apply_patch_header_chars(paths: Sequence[Path | str], *, operation: str = "Update") -> int:
    """Estimate apply_patch header characters for one multi-file patch."""
    total = len("*** Begin Patch\n") + len("*** End Patch\n")
    total += sum(estimate_apply_patch_target_chars(path, operation=operation) for path in paths)
    return total


# 関数: `partition_paths_by_budget` の入出力契約と処理意図を定義する。

def partition_paths_by_budget(
    paths: Sequence[Path | str],
    *,
    max_entries: int | None = None,
    max_chars: int | None = None,
    base_chars: int = 0,
    item_weight: Callable[[Path | str], int] | None = None,
) -> list[list[Path]]:
    """Partition paths into stable batches constrained by count and character budgets."""
    batches: list[list[Path]] = []
    current_batch: list[Path] = []
    current_chars = int(base_chars)
    entry_limit = None if max_entries is None or int(max_entries) <= 0 else int(max_entries)
    char_limit = None if max_chars is None or int(max_chars) <= 0 else int(max_chars)
    weight_fn = item_weight or (lambda candidate: len(normalize_repo_relative_path(candidate)))

    for raw_path in paths:
        candidate = Path(raw_path)
        candidate_weight = int(weight_fn(raw_path))
        would_exceed_entries = entry_limit is not None and current_batch and len(current_batch) >= entry_limit
        would_exceed_chars = char_limit is not None and current_batch and (current_chars + candidate_weight) > char_limit
        if would_exceed_entries or would_exceed_chars:
            batches.append(current_batch)
            current_batch = []
            current_chars = int(base_chars)

        current_batch.append(candidate)
        current_chars += candidate_weight

    if current_batch:
        batches.append(current_batch)

    return batches


# 関数: `load_path_manifest` の入出力契約と処理意図を定義する。

def load_path_manifest(path: Path) -> list[Path]:
    """Load one newline-separated path manifest.

    Rules:
    - UTF-8 text
    - blank lines ignored
    - lines starting with `#` ignored
    - relative entries resolved against the manifest directory when present
    - otherwise relative entries fall back to the repo root
    """
    lines = path.read_text(encoding="utf-8").splitlines()
    resolved: list[Path] = []
    for raw_line in lines:
        stripped = raw_line.strip()
        # 条件分岐: `not stripped or stripped.startswith("#")` を満たす経路を評価する。
        if not stripped or stripped.startswith("#"):
            continue

        candidate = Path(stripped)
        # 条件分岐: `not candidate.is_absolute()` を満たす経路を評価する。
        if not candidate.is_absolute():
            manifest_relative = (path.parent / candidate)
            repo_relative = (ROOT / candidate)
            if manifest_relative.exists():
                candidate = manifest_relative.resolve()
            elif repo_relative.exists():
                candidate = repo_relative.resolve()
            else:
                candidate = manifest_relative.resolve()

        resolved.append(candidate)

    return resolved


# 関数: `expand_cli_paths` の入出力契約と処理意図を定義する。

def expand_cli_paths(paths: Sequence[str] | None, path_files: Sequence[str] | None = None) -> list[Path]:
    """Expand direct path arguments plus `@manifest` and `--paths-file` inputs."""
    ordered: list[Path] = []
    seen: set[str] = set()

    # 関数: `add_path` の入出力契約と処理意図を定義する。
    def add_path(candidate: Path) -> None:
        key = str(candidate)
        # 条件分岐: `key not in seen` を満たす経路を評価する。
        if key not in seen:
            seen.add(key)
            ordered.append(candidate)

    for raw_path in paths or []:
        # 条件分岐: `raw_path.startswith("@")` を満たす経路を評価する。
        if raw_path.startswith("@"):
            for manifest_path in load_path_manifest(Path(raw_path[1:]).resolve()):
                add_path(manifest_path)

            continue

        add_path(Path(raw_path))

    for raw_manifest in path_files or []:
        for manifest_path in load_path_manifest(Path(raw_manifest).resolve()):
            add_path(manifest_path)

    return ordered


# 関数: `summarize_longest_paths` の入出力契約と処理意図を定義する。

def summarize_longest_paths(paths: Iterable[Path], *, top_n: int = 10) -> list[tuple[int, int, str]]:
    """Return the longest paths as `(path_len, name_len, path_text)` tuples."""
    ranked = sorted(
        ((len(str(path)), len(path.name), str(path)) for path in paths),
        reverse=True,
    )
    return ranked[: int(top_n)]
