from __future__ import annotations

import re
from pathlib import Path


def _split_history_blocks(text: str) -> tuple[str, list[str]]:
    match = re.search(r"^##\s+20\d\d-\d\d-\d\d", text, flags=re.M)
    # 条件分岐: `not match` を満たす経路を評価する。
    if not match:
        return text, []

    header = text[: match.start()].rstrip() + "\n\n"

    starts = [m.start() for m in re.finditer(r"^##\s+20\d\d-\d\d-\d\d", text, flags=re.M)]
    starts.append(len(text))

    blocks: list[str] = []
    for start, end in zip(starts, starts[1:]):
        block = text[start:end].strip("\n")
        # 条件分岐: `not block` を満たす経路を評価する。
        if not block:
            continue

        lines = block.splitlines()
        while lines and lines[-1].strip() == "":
            lines.pop()

        while lines and lines[-1].strip() == "---":
            lines.pop()
            while lines and lines[-1].strip() == "":
                lines.pop()

        blocks.append("\n".join(lines).rstrip() + "\n")

    return header, blocks


def _make_recent_header() -> str:
    return (
        "# WORK HISTORY RECENT\n"
        "This file starts with an ASCII-only preamble to avoid a known Windows tool unicode slicing issue.\n"
        "Japanese content begins after this preamble.\n"
        "\n"
        "# 作業履歴（直近3ログ）\n"
        "\n"
        "このファイルは AI が作業開始時に参照する **直近3ログ**（`doc/WORK_HISTORY.md` から自動抽出）である。\n"
        "完全な履歴は `doc/WORK_HISTORY.md` を参照する。\n"
        "\n"
        "---\n"
        "\n"
    )


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    src = root / "doc" / "WORK_HISTORY.md"
    dst = root / "doc" / "WORK_HISTORY_RECENT.md"

    text = src.read_text(encoding="utf-8")
    _, blocks = _split_history_blocks(text)
    recent = blocks[-3:]

    content = _make_recent_header()
    # 条件分岐: `recent` を満たす経路を評価する。
    if recent:
        content += "\n\n---\n\n".join(block.strip() + "\n" for block in recent).rstrip() + "\n"

    dst.write_text(content, encoding="utf-8")
    print(f"[ok] wrote: {dst}")


# 条件分岐: `__name__ == "__main__"` を満たす経路を評価する。

if __name__ == "__main__":
    main()

