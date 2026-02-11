# 論文化（Phase 8 / Step 8.2）: 作業フォルダ

目的：
- `output/` に固定された図表・数値を参照しながら、投稿可能な原稿（まずはMarkdown）を組み立てる。

方針：
- 本文はまず `doc/paper/00_outline.md` をベースに、章ごとに肉付けする。
- 図表の参照先は “ファイルパス固定” を優先する（図の差し替えが容易になる）。
- 先行研究との比較（GR/PPN/実験）と、「一致する範囲／差が出る範囲（強場・宇宙論）」を明確に書く。

入口：
- 資料生成（軽量；HTMLのみ）：`cmd /c output\summary\build_materials.bat quick-nodocx`
- 資料生成（Full；重い）：`cmd /c output\summary\build_materials.bat`
- 論文ビルド入口（Part I；Table 1＋論文HTML＋lint；HTMLのみ）：`python -B scripts/summary/paper_build.py --profile paper --mode publish --outdir output/summary --skip-docx`
- 論文ビルド入口（Part II；HTMLのみ）：`python -B scripts/summary/paper_build.py --profile part2_astrophysics --mode publish --outdir output/summary --skip-docx --skip-tables`
- 論文ビルド入口（Part III；HTMLのみ）：`python -B scripts/summary/paper_build.py --profile part3_quantum --mode publish --outdir output/summary --skip-docx --skip-tables`
- 短報ビルド入口（Table 1＋短報HTML＋lint；HTMLのみ）：`python -B scripts/summary/paper_build.py --profile short_note --mode publish --outdir output/summary --skip-docx`
- 本文アウトライン：`doc/paper/00_outline.md`
- 図表インデックス：`doc/paper/01_figures_index.md`
- 本文（Part I：コア理論）：`doc/paper/10_part1_core_theory.md`
- 本文（Part II：宇宙物理）：`doc/paper/11_part2_astrophysics.md`
- 本文（Part III：量子物理）：`doc/paper/12_part3_quantum.md`
- 短報（short note）：`doc/paper/15_short_note.md`
- （旧：統合版草稿）：`doc/paper/10_manuscript.md`（参照用）
- Table 1（自動生成）：`output/summary/paper_table1_results.md`（生成: `cmd /c output\summary\build_materials.bat quick-nodocx`）
- 一般向け統一レポート（図のカタログとして使える）：`output/summary/pmodel_public_report.html`
- 参考文献：`doc/paper/30_references.md`
