# v3 Trial-1 nucleon-scale Phase 1 inventory

Date: 2026-04-23

## Purpose

This note closes Phase 1 of the v3 Trial-1 baryon mass-ratio / proton-radius
workflow. Phase 1 does not run the heavy nucleon scan. It fixes the
paper-critical dependency surface and the minimal modification contract that
Phase 2 must obey.

## Result

The inventory output is:

- `output/public/quantum/v3_trial1_nucleon_scale_phase1_inventory.json`
- `output/public/quantum/v3_trial1_nucleon_scale_phase1_inventory.csv`

Gate result:

- `overall_status=ready_for_phase2_scan_implementation`
- `can_advance_to_phase2_scan_implementation=true`

The retained W/Z and Trial-3 contract is still present:

- localized sectors: `41`
- exact vector rows: `905838`
- `kappa_W=1.6024037199246677`
- `kappa_Z=1.6817234303593958`
- base modes: `324937`

The reusable source surfaces required by the next scan were found in the
restored scripts:

- `solve_sector_profile`, `cached_profile`, `scan_ell_sector`
- `interpolate_integer_modes`, `build_base_modes`
- `polarization_weight`, `coupled_charge_factor`, `coupled_mass_factor`
- `build_exact_ladder`
- `solve_two_component_profile`
- `interpolate_two_component_modes`, `best_ratio_pair_fast`
- `normalize_vector_rows`, `best_ratio_pair`

## Minimal Modification Contract

Allowed for Phase 2:

- retarget the absolute energy scale from W/Z to the nucleon scale
- expand family labels for the nucleon search
- keep or refine the spatial grid so the proton-radius target is resolved

Prohibited for Phase 2:

- adding new independent model constants
- adding new coupling or potential terms
- relaxing the mass-ratio or radius target
- inserting an intermediate boundary-condition target
- rewriting the restored W/Z solver chain invasively

The current dimensionless grid is sufficient for the first nucleon-scale
radius audit:

- nucleon length: `0.21030893003308207 fm`
- `0.84 fm` target in dimensionless units: `3.994124262188325`
- current `R_MAX=25` at nucleon scale: `5.257723250827052 fm`
- current `MAX_STEP=0.10` at nucleon scale: `0.02103089300330821 fm`

## Push Scope

The inventory writes an explicit `dependency_list`. For the next Trial-1 commit,
only files listed there are paper-critical by default. The 80 other restored
`output/public/quantum` metrics remain local holding artifacts recorded by
`output/private/summary/windows_worksets/v3_trial1_untracked_public_quantum_metrics_20260423.csv`.

This branch is not a theory-stuck branch. No roadmap rewrite is needed yet.
If the Phase 2 broad scan fails to find an admissible neutron/proton ratio and
proton radius candidate under the minimal contract, the blocker must be
classified explicitly as computation, model-surface, or theory-ansatz failure
before the roadmap is reorganized.

## Reproduction

```powershell
python -B scripts\quantum\v3_trial1_nucleon_scale_phase1_inventory.py
python -B scripts\summary\enforce_python_block_spacing.py --paths scripts\quantum\v3_trial1_nucleon_scale_phase1_inventory.py
python -B scripts\summary\enforce_python_def_class_comments.py --paths scripts\quantum\v3_trial1_nucleon_scale_phase1_inventory.py
python -m py_compile scripts\quantum\v3_trial1_nucleon_scale_phase1_inventory.py
python -m json.tool output\public\quantum\v3_trial1_nucleon_scale_phase1_inventory.json
```

## Next Work Order

1. Implement Phase 2 broad scan using only the listed W/Z and two-component
   solver chain.
2. Score candidates simultaneously against `m_n/m_p=1.001378...` and
   `r_p ~= 0.84 fm`.
3. Record candidate tables and rejection reasons in fixed public JSON/CSV
   outputs.
4. If no candidate survives, run the retry-branch gate and decide whether the
   failure is computational coverage or a theory-ansatz blocker before changing
   the roadmap.
