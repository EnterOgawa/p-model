# STATUS

- Current focus: v3.0 challenge handoff after the v2.0 bilingual release completion.
- Status: v2.0 JP/EN papers, figure corrections, README synchronization, and GitHub publication updates are complete and pushed.
- Result:
  - English paper series is closed after the repeated third-party audit fixes and the final Part V EN Final Note cleanup.
  - Root `README.md` and `output/public/README.md` are synchronized for the v2.0 bilingual release.
  - Latest pushed commit: `250ed757` (`Polish README wording`) on `origin/main`.
  - Previous completion commits: `c04ea8c0` (`Finalize English paper revisions`), `c48821f7` (`Update public README for bilingual release`), and `59818b84` (`Sync root README for bilingual release`).
- Evidence:
  - [README.md](/C:/develop/waveP/README.md)
  - [output/public/README.md](/C:/develop/waveP/output/public/README.md)
  - [pmodel_paper_en.pdf](/C:/develop/waveP/papers/locales/en/pmodel_paper_en.pdf)
  - [pmodel_paper.pdf](/C:/develop/waveP/papers/pmodel_paper.pdf)
  - [pmodel_paper_part5_future_predictions_en.pdf](/C:/develop/waveP/papers/locales/en/pmodel_paper_part5_future_predictions_en.pdf)
- Verification:
  - Final English paper branch: `paper_tex_audit`, `paper_pdf`, and `paper_locale_qc` were `ok=True` for Part V EN and full paper EN.
  - README branch: root README was copied from the user-updated public README, then `sync_public_readme.py --direction root-to-public` was run before commit/push.
  - GitHub push completed through `250ed757` on `main`.
- Environment:
  - Default work mode: Windows native in `C:\develop\waveP`.
  - RHEL 9.2 VM is available only when needed for Corrfunc/Linux-dependent computation.
  - RHEL details: host `192.100.1.111`, user `mcp7`, repo `/home/mcp7/work/waveP`, venv `/home/mcp7/work/waveP/.venv_wsl`.
- Next:
  1. Start v3.0 from a fresh task definition after context clear.
  2. At session start, read `doc/AI_CONTEXT_MIN.json`, `doc/STATUS.md`, `doc/ROADMAP.md`, `doc/WORK_HISTORY_RECENT.md`, and `doc/PRIMARY_SOURCES.md`.
  3. If v3.0 requires Corrfunc or other Linux-only work, ask the user to start the RHEL 9.2 VM first.
