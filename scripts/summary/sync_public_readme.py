from __future__ import annotations

import argparse
import hashlib
import shutil
from pathlib import Path


# 関数: `_sha256` の入出力契約と処理意図を定義する。
def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# 関数: `_copy_if_needed` の入出力契約と処理意図を定義する。

def _copy_if_needed(src: Path, dst: Path, dry_run: bool = False) -> bool:
    # 条件分岐: `not src.exists()` を満たす経路を評価する。
    if not src.exists():
        raise FileNotFoundError(f"missing source: {src}")

    # 条件分岐: `dst.exists() and _sha256(src) == _sha256(dst)` を満たす経路を評価する。

    if dst.exists() and _sha256(src) == _sha256(dst):
        print(f"[ok] up-to-date: {dst}")
        return False

    # 条件分岐: `dry_run` を満たす経路を評価する。

    if dry_run:
        print(f"[dry-run] copy: {src} -> {dst}")
        return True

    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    print(f"[ok] copied: {src} -> {dst}")
    return True


# 関数: `main` の入出力契約と処理意図を定義する。

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Sync README between output/public and repo root."
    )
    parser.add_argument(
        "--direction",
        choices=("public-to-root", "root-to-public"),
        default="public-to-root",
        help="Sync direction (default: public-to-root).",
    )
    parser.add_argument(
        "--bootstrap",
        action="store_true",
        help=(
            "When direction is public-to-root and output/public/README.md is missing, "
            "initialize it from root README.md before syncing."
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    repo_readme = root / "README.md"
    public_readme = root / "output" / "public" / "README.md"

    # 条件分岐: `args.direction == "public-to-root"` を満たす経路を評価する。
    if args.direction == "public-to-root":
        # 条件分岐: `not public_readme.exists()` を満たす経路を評価する。
        if not public_readme.exists():
            # 条件分岐: `not args.bootstrap` を満たす経路を評価する。
            if not args.bootstrap:
                print(f"[err] missing: {public_readme}")
                print("[hint] run with --bootstrap once to initialize from root README.")
                return 1

            # 条件分岐: `not repo_readme.exists()` を満たす経路を評価する。

            if not repo_readme.exists():
                print(f"[err] missing bootstrap source: {repo_readme}")
                return 1

            _copy_if_needed(repo_readme, public_readme, dry_run=args.dry_run)

        _copy_if_needed(public_readme, repo_readme, dry_run=args.dry_run)
        return 0

    # 条件分岐: `not repo_readme.exists()` を満たす経路を評価する。

    if not repo_readme.exists():
        print(f"[err] missing source: {repo_readme}")
        return 1

    _copy_if_needed(repo_readme, public_readme, dry_run=args.dry_run)
    return 0


# 条件分岐: `__name__ == "__main__"` を満たす経路を評価する。

if __name__ == "__main__":
    raise SystemExit(main())
