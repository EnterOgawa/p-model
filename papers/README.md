# papers

This directory stores publish-ready paper PDFs generated from TeX.

Generation command (all profiles):

```bat
python -B scripts/summary/paper_pdf.py --profile paper --profile part2_astrophysics --profile part3_quantum --profile part4_verification --outdir output/private/summary --engine auto --require-engine --sync-papers --papers-dir papers
```

Typical workflow (HTML/TeX audit + PDF sync):

```bat
cmd /c scripts\summary\build_materials.bat quick-nodocx
```
