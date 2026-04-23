# v3 Trial-1 Push Scope Cleanup

Date UTC: 2026-04-23T13:24:32Z

## Context

User correction:

> push はその計算が論文を構成するうえで必要なものだけでよい

This cleanup fixes the operating rule before Phase 1. The target is not to
push every recovered script or every restored artifact just because it exists.
The target is to preserve and push only the computation chain required to
rebuild paper-facing scientific surfaces.

## Definitions

`paper-critical` means a script or public artifact that regenerates one of:

- a numerical value written in the paper,
- a figure used by the paper,
- a validation / scoreboard table used by the paper,
- a public metrics JSON/CSV directly cited by those paper surfaces.

`holding` means a recovered or historical calculation that is useful locally
but is not yet part of the active paper dependency chain.

`excluded by default` means exploratory, failed-route, archive-only, or
abandoned branch code. These files are not committed or pushed unless a later
paper dependency map promotes them.

## Current State

Known recent commits:

- `86a8d48d` restored the direct Trial-1 Phase 0 W/Z reproduction scripts.
  This set is paper-critical for the current Phase 0 gate.
- `fb7ed024` restored 2,152 stashed quantum scripts.
  This broad set is not treated as automatically paper-critical.
- `b8e78baa` tracked 20 Cassini/Mercury scripts and the `.gitignore` fix.
  This broad set is not treated as automatically paper-critical for Trial-1.

No history rewrite is done here. Removing already-pushed files from tracking
requires a separate dependency audit, because many historical roadmap surfaces
refer to generated metrics from these branches. A blind mass removal could
damage reproducibility of existing paper or roadmap results.

## Untracked Public Metrics

There are 80 untracked files under `output/public/quantum/` after the Phase 0
restore/recalculation. They are not staged in this cleanup.

Private hash manifest:

`output/private/summary/windows_worksets/v3_trial1_untracked_public_quantum_metrics_20260423.csv`

Related private script manifests:

- `output/private/summary/windows_worksets/86a8d48d_trial1_wz_scripts_manifest_20260423.txt`
- `output/private/summary/windows_worksets/fb7ed024_restored_scripts_manifest_20260423.txt`
- `output/private/summary/windows_worksets/b8e78baa_cassini_mercury_manifest_20260423.txt`

## Rule From This Point

Before the next Trial-1 commit/push, the active calculation must emit or update
an explicit dependency list with these fields:

- `paper_critical_scripts`
- `paper_critical_public_outputs`
- `holding_local_only`
- `excluded_by_default`
- `reason`

Only `paper_critical_scripts` and required `paper_critical_public_outputs` may
be staged. Broad existence-based staging is prohibited.

## Next Step

Phase 1 should start with a nucleon-scale inventory script that maps the W/Z
solver components into a minimal paper-critical dependency list for the
`m_n/m_p` and proton-radius calculation. No new broad restore or broad artifact
push should be performed before that list exists.
