# P-model (Time‑Wave Density **P**)

This repository is the **verification workspace** for the **P-model** research program.
It is organized to make the model **auditable, reproducible, and falsifiable** via scripts and fixed outputs.

## The idea (story, not equations)

P-model starts from a simple narrative:

- **Time is not only a coordinate** used to label events; it is treated as a **physical wave field**.
- The local “density” (or intensity) of that time‑wave is denoted by **P**.
- **Matter is wave**: stable objects are interpreted as **bound / standing‑wave structures** in this field.
- What we call “gravity” is modeled operationally as **how P varies in space**, and how that variation affects
  - clock rates (what a clock counts),
  - signal propagation (how light/radio travels),
  - and the dynamics inferred from those measurements.

The goal is **not** to dismiss GR/SR as practical effective theories.
Instead, P-model aims to **re-derive the same observational successes from a different mechanism** and then
isolate **where strong predictions diverge** (so the model can be rejected if nature disagrees).

## What this GitHub repository is for

- **Scripts**: reproducible analysis code (the primary “methods”).
- **Verification outputs**: fixed artifacts (JSON/CSV/PNG) that are intended to be cited by manuscripts.

Documents such as full manuscripts are **published separately**.
This repo is intended to stay focused on **verification scripts and the cited outputs they produce**.

## Notes

- Large raw inputs are not stored here; scripts are expected to **fetch / cache / preprocess** as needed.
- Claims should be evaluated against the **fixed outputs** and **explicit falsification criteria** (not narratives).
