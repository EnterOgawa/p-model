# P-model (Time‑Wave Density **P**)

This repository hosts the manuscripts and reproducible analyses for the **P-model** research program.
The core working hypothesis is **“matter = wave”** and **time is a physical wave field** whose local density is denoted by **P**.

The goal is *not* to dismiss GR/SR as practical effective theories, but to **re-derive the same observational successes from an alternative mechanism**, while keeping the project **auditable, reproducible, and falsifiable**.  
Where GR/SR and P-model agree, the agreement is treated as an expected weak-field/operational coincidence; where they differ, the difference is formulated as a testable prediction.

## Key conventions (signs / definitions)

- **P increases near mass** (P > P0).
- Newton-like potential: **φ ≡ −c² ln(P/P0)** (φ → 0 at infinity).
- Gravitational acceleration: **a = −∇φ = c² ∇ln(P/P0)**.
- Light propagation uses **n(P) = (P/P0)^(2β)** (PPN mapping: (1+γ) = 2β).

Canonical definitions and sign conventions live in `doc/P_model_handoff.md`.

## What is in this repo

- **Manuscripts** (Part I–III) under `doc/paper/`
- **Reproducible scripts** under `scripts/`
- **Operational roadmap & logs**: `doc/ROADMAP.md`, `doc/STATUS.md`, `doc/WORK_HISTORY.md`
- **Primary sources index** (URLs + accessed dates + caching policy): `doc/PRIMARY_SOURCES.md`

## Reproducibility / verification materials

The published manuscripts intentionally avoid internal file paths and script names to keep the narrative readable.
Reproducibility details (fixed artifacts, audit gates, hashes, and minimal rebuild commands) are collected in:

- `doc/verification/VERIFICATION_MATERIALS.md`

Publishing workflow (HTML/DOCX generation, QC, release bundles) is summarized in:

- `doc/PUBLISHING.md`

## Notes

- Large raw inputs and large generated outputs are generally **not intended for normal Git commits**; they should be attached as **release artifacts** when needed.
- This is an active research repository. Claims should be evaluated against the fixed outputs, audit gates, and falsification criteria.

