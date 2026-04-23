# STATUS

- Current focus: `part5_en` Final Note wording/typesetting cleanup on the canonical paper surface.
- Status: live Part V EN Final Note review note fixed on the canonical paper surface.
- Result:
  - Part V EN Final Note now uses `as we have known it`, `Everything began with`, `Science too is something done by human beings.`, proper LaTeX opening/closing quotes, and `we are sustained by the people around us`.
  - `the singularity` was kept intentionally to preserve the JP original's assertive meaning.
  - Part V EN and full EN paper were rebuilt after the fixed-content script update.
- Evidence:
  - Final Note source block:
    - [paper_profile_content.py](/C:/develop/waveP/scripts/summary/paper_profile_content.py)
  - canonical EN Part V paper:
    - [pmodel_paper_part5_future_predictions_en.pdf](/C:/develop/waveP/papers/locales/en/pmodel_paper_part5_future_predictions_en.pdf)
  - canonical EN full paper:
    - [pmodel_paper_en.pdf](/C:/develop/waveP/papers/locales/en/pmodel_paper_en.pdf)
  - Part V EN final-note text probe:
    - [part5_en_final_note_check.txt](/C:/develop/waveP/output/private/summary/locales/en/part5_en_final_note_check.txt)
  - actual Part V EN final-note page:
    - [page-25.png](/C:/develop/waveP/output/private/summary/page_audit/part5_en_final_note_review/page-25.png)
  - actual full EN final-note page:
    - [page-53.png](/C:/develop/waveP/output/private/summary/page_audit/paper_en_final_note_review/page-53.png)
- Verification:
  - `paper_tex_audit --profile part5_future_predictions --locale en --engine xelatex`: `ok=True`
  - `paper_pdf --profile part5_future_predictions --locale en --engine xelatex --sync-papers --papers-dir papers`: `ok=True warnings=0`
  - `paper_locale_qc --profile part5_future_predictions --locale en --mode all`: `ok=True errors=0 warnings=0`
  - `paper_tex_audit --profile paper --locale en --engine xelatex`: `ok=True`
  - `paper_pdf --profile paper --locale en --engine xelatex --sync-papers --papers-dir papers`: `ok=True warnings=0`
  - `paper_locale_qc --profile paper --locale en --mode all`: `ok=True errors=0 warnings=0`
- Root cause:
  - The Final Note was not sourced from `doc/paper/locales/en/14_part5_future_predictions.md`; it was injected from the fixed post-bibliography content block in `scripts/summary/paper_profile_content.py`.
  - Earlier inspection of the markdown file alone would not have caught these live English wording issues.
- Next:
  1. Treat Part V EN Final Note as closed unless a new paper-surface issue appears on page 25 / full-paper page 53.
  2. For future Final Note revisions, inspect `paper_profile_content.py` first, then rebuild `part5_future_predictions` and full `paper` EN together.
