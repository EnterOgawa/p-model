# STATUS

- Current focus: v3.0 Trial-1 baryon mass-ratio / proton-radius challenge, Phase 1 nucleon-scale dependency inventory.
- Status: Phase 1 is complete. `overall_status=ready_for_phase2_scan_implementation`, `can_advance_to_phase2_scan_implementation=true`.
- Roadmap state:
  - Phase 0 readiness restoration: closed.
  - Phase 1 paper-critical dependency inventory: closed.
  - Phase 2 broad nucleon-scale scan: next.
  - No theory-stuck branch was reached in this turn, so the roadmap was not reorganized.
- New script:
  - `scripts/quantum/v3_trial1_nucleon_scale_phase1_inventory.py`
- New note:
  - `doc/quantum/188_v3_trial1_nucleon_scale_phase1_inventory.md`
- New outputs:
  - `output/public/quantum/v3_trial1_nucleon_scale_phase1_inventory.json`
  - `output/public/quantum/v3_trial1_nucleon_scale_phase1_inventory.csv`
- Phase 1 gate results:
  - W/Z archive assets: `pass`
  - reusable solver functions: `pass`
  - minimal modification contract: `pass`
  - nucleon-scale radius grid resolution: `pass`
  - paper-critical dependency list: `pass`
- Retained W/Z / Trial-3 checks:
  - localized sectors `41`
  - exact vector rows `905838`
  - `kappa_W=1.6024037199246677`
  - `kappa_Z=1.6817234303593958`
  - base modes `324937`
- Nucleon-scale diagnostics:
  - proton mass target for scoring: `938.272 MeV`
  - scale factor from W to nucleon mass: `0.011674551132899502`
  - nucleon length: `0.21030893003308207 fm`
  - `0.84 fm` target in dimensionless units: `3.994124262188325`
  - current `R_MAX=25` at nucleon scale: `5.257723250827052 fm`
  - current `MAX_STEP=0.10` at nucleon scale: `0.02103089300330821 fm`
- Minimal modification contract for Phase 2:
  - Allowed: absolute energy-scale retargeting, family-range expansion, grid refinement.
  - Prohibited: new independent constants, new coupling/potential terms, target relaxation, intermediate boundary-condition targets, invasive W/Z solver rewrite.
- Paper-critical push scope:
  - The Phase 1 JSON contains `dependency_list.paper_critical_scripts` and `dependency_list.paper_critical_public_outputs`.
  - Only those files are eligible for the next Trial-1 commit/push by default.
  - The remaining 80 restored `output/public/quantum` metrics remain local holding artifacts in `output/private/summary/windows_worksets/v3_trial1_untracked_public_quantum_metrics_20260423.csv`.
- Verification completed:
  - `python -B scripts/summary/enforce_python_block_spacing.py --paths scripts/quantum/v3_trial1_nucleon_scale_phase1_inventory.py --fix`
  - `python -B scripts/summary/enforce_python_def_class_comments.py --paths scripts/quantum/v3_trial1_nucleon_scale_phase1_inventory.py --fix`
  - `python -B scripts/summary/enforce_python_block_spacing.py --paths scripts/quantum/v3_trial1_nucleon_scale_phase1_inventory.py`
  - `python -B scripts/summary/enforce_python_def_class_comments.py --paths scripts/quantum/v3_trial1_nucleon_scale_phase1_inventory.py`
  - `python -m py_compile scripts/quantum/v3_trial1_nucleon_scale_phase1_inventory.py`
  - `python -B scripts/quantum/v3_trial1_nucleon_scale_phase1_inventory.py`
  - `python -m json.tool output/public/quantum/v3_trial1_nucleon_scale_phase1_inventory.json`
- Warnings:
  - This Phase 1 step does not run the nucleon scan; it only freezes the adapter and dependency contract.
  - The external workflow note under `C:\Users\ogawa\Downloads\` was read as user-supplied context, but repo-local scripts and outputs are now the reproducible surface.
  - Paper build was not run, per the rule requiring explicit user instruction.
- Next:
  1. Implement `scripts/quantum/v3_trial1_nucleon_scale_phase2_scan.py` or the repo-local equivalent by reusing the listed W/Z / two-component solver chain.
  2. Run the broad scan against both `m_n/m_p=1.001378...` and `r_p ~= 0.84 fm`.
  3. Write fixed public JSON/CSV candidate and rejection outputs.
  4. If no admissible candidate survives, classify the blocker as computation coverage, model-surface limitation, or theory-ansatz failure before changing the roadmap.
