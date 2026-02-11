# 図表インデックス（固定パス）

このインデックスは「本文から参照する図表のファイルパス」を固定するための一覧。
図を差し替えても本文側の参照が壊れないように、原則としてパスは変更しない。

注記：`output/summary/pmodel_paper.html` の図番号（図1, 図2, …）は、本文（doc/paper/*.md）での **最初の参照順（読む順）** で自動付与される。  
このファイルは「固定パス一覧（参照を壊さない）」と「captionの元」を正とする。図番号を固定したい段階では、本文側の参照順（どこで最初に出すか）を整理する。

入口（カタログ）：
- `output/summary/pmodel_public_report.html`

---

## 共通（一般向け統一レポート）

- `output/summary/pmodel_public_report.html`（全章）

---

## Table（論文用）

- `output/summary/paper_table1_results.md`（Table 1: 検証サマリ）
- `output/summary/paper_table1_results.csv`（数値）
- `output/summary/paper_table1_results.json`（メタデータ）
- `output/summary/pmodel_paper.html`（論文HTML: 本文（Markdown）＋Table 1＋図表一覧）

---

## コア（Part I）

- `output/theory/pmodel_core_mapping_overview.png`（P-model写像概念図：P→φ→重力/時計/光、背景P（宇宙論）と局所P構造（量子））
- `output/theory/pmodel_core_concept_comparison.png`（P-modelと参照枠（GR）の概念比較（重力/運動/光/時間/赤方偏移））

---

## 決定的検証（Phase 7）

- `output/theory/frozen_parameters.json`（βの凍結：一次ソースで固定）
- `output/summary/validation_scoreboard.png`（総合スコアボード：全検証の現状（色分け：OK/要改善/不一致））
- `output/summary/validation_scoreboard.json`（数値＋判定基準（policy）＋Table 1 の行別判定）
- `output/summary/quantum_scoreboard.png`（総合スコアボード（量子）：Part III Table 1 の俯瞰（色分け：OK/要改善/不一致））
- `output/summary/quantum_scoreboard.json`（数値（proxy）＋判定基準（policy））
- `output/summary/decisive_falsification.png`（反証条件パック：必要精度と棄却条件の要点）
- `output/summary/decisive_falsification.json`（数値）

---

## 量子（粒子＝束縛波）

- `output/quantum/particle_reflection_demo.png`（粒子＝反射／境界条件：波束の反射と最小の束縛モードのデモ）

---

## 量子（核力・束縛エネルギー）

- `output/quantum/nuclear_near_field_interference_two_mode_model.png`（図A：近接場干渉の概念図。`Δω=2J(R0)` と `B.E.=ħΔω` の最小I/Fを固定（Step 7.13.19.12））
- `output/quantum/nuclear_binding_energy_frequency_mapping_deuteron_two_body.png`（図B：重陽子2体系の準位/境界条件図。`k cot(kR)=-κ` を満たす代表半径のスケールを固定（Step 7.13.19.12））
- `output/quantum/nuclear_binding_energy_frequency_mapping_deuteron_verification.png`（重陽子（H-2）の B.E. 理論値と観測値の一致度、相対誤差と残差要因を固定（Step 7.13.19.5））
- `output/quantum/nuclear_binding_energy_frequency_mapping_alpha_verification.png`（He-4 の多体系検証（pair-wise / collective）で B.E./A の系統差を固定（Step 7.13.19.6））
- `output/quantum/nuclear_binding_energy_frequency_mapping_representative_nuclei.png`（軽核代表（A=2–16）での B.E./A vs A と残差パターン（偶奇・magic近傍）を固定（Step 7.13.19.7））
- `output/quantum/nuclear_binding_energy_frequency_mapping_ame2020_all_nuclei.png`（図C：AME2020全核種での理論vs観測曲線と残差基準線（Step 7.13.19.12））
- `output/quantum/nuclear_binding_energy_frequency_mapping_differential_predictions.png`（図D：距離proxy差し替え（global R vs local d）による残差監査（Step 7.13.19.12））
- `output/quantum/nuclear_binding_energy_frequency_mapping_differential_quantification.png`（図E：差分 `DeltaB` と必要精度 `σ_req=abs(DeltaB)/3` の分布固定（Step 7.13.19.12））
- `output/quantum/nuclear_binding_energy_frequency_mapping_minimal_additional_physics.png`（核結合エネルギー Δω→B.E. 写像：local spacing d と ν飽和の比較。凍結した 3σ 判定で pass/fail がどう変わるかを固定（Step 7.13.18））
- `output/quantum/nuclear_binding_energy_frequency_mapping_theory_diff.png`（核結合エネルギー Δω→B.E. 写像：P-model（local+d, ν飽和）と標準理論proxy（湯川距離proxy/SEMF固定係数）の差分監査（Step 7.13.17.11））
- `output/quantum/nuclear_binding_energy_frequency_mapping_falsification_pack.png`（核結合エネルギー Δω→B.E. 写像：反証条件パック（z_median・A-trend）に独立cross-check（分離エネルギー/電荷半径kink）を統合した運用ゲート（Step 7.13.17.14））

---

## 量子（測定：Born則・状態更新）

- `output/quantum/quantum_measurement_born_rule_flow.png`（Born則（|ψ|²）と状態更新（条件付き更新）の位置づけ図（Step 7.10））
- `output/quantum/quantum_measurement_born_rule_metrics.json`（要点）

---

## 量子（電磁気：電荷・Maxwell/光子）

- `output/quantum/electromagnetism_minimal.png`（Coulomb 1/r と光子（EM波）の最小位置づけ：スケール確認＋仮定列の固定（Step 7.11））
- `output/quantum/electromagnetism_minimal_metrics.json`（数値）
- `output/quantum/alpha_p_dependence_constraint.png`（α(P) の κ 制約：原子時計（Rosenband）＋分光（H 1S–2S）の order constraint を同一図で固定（Step 7.11））
- `output/quantum/alpha_p_dependence_constraint.json`（数値＋仮定列）

---

## 量子（原子・分子）

- `output/quantum/atomic_hydrogen_baseline.png`（原子スペクトル基準値（H I；NIST ASD）：Lyα/Hα/Hβ/Hγ の vacuum wavelength を固定（Step 7.12））
- `output/quantum/atomic_hydrogen_baseline_metrics.json`（数値）
- `output/quantum/atomic_hydrogen_hyperfine_baseline.png`（原子スペクトル基準値（H I hyperfine；NIST AtSpec）：21 cm 線（基底 hyperfine）を固定（Step 7.12））
- `output/quantum/atomic_hydrogen_hyperfine_baseline_metrics.json`（数値）
- `output/quantum/atomic_helium_baseline.png`（原子スペクトル基準値（He I；NIST ASD）：多電子系の最小として 4本の vacuum wavelength を固定（Step 7.12））
- `output/quantum/atomic_helium_baseline_metrics.json`（数値）
- `output/quantum/molecular_h2_baseline.png`（分子定数基準値（H2；NIST WebBook）：ωe/ωexe/Be/αe/D(dist)/re を固定（Step 7.12））
- `output/quantum/molecular_h2_baseline_metrics.json`（数値）
- `output/quantum/molecular_hd_baseline.png`（分子定数基準値（HD；NIST WebBook）：同位体依存（縮約質量スケール）チェック用（Step 7.12））
- `output/quantum/molecular_hd_baseline_metrics.json`（数値）
- `output/quantum/molecular_d2_baseline.png`（分子定数基準値（D2；NIST WebBook）：同位体依存（縮約質量スケール）チェック用（Step 7.12））
- `output/quantum/molecular_d2_baseline_metrics.json`（数値）
- `output/quantum/molecular_isotopic_scaling.png`（同位体依存チェック：ωe∝μ^{-1/2}, B_e∝μ^{-1} の ratio（Step 7.12））
- `output/quantum/molecular_isotopic_scaling_metrics.json`（数値）
- `output/quantum/molecular_dissociation_thermochemistry.png`（解離エンタルピー（298 K；NIST WebBook thermochemistry）：結合エネルギーの独立ベースライン（Step 7.12））
- `output/quantum/molecular_dissociation_thermochemistry_metrics.json`（数値）
- `output/quantum/molecular_dissociation_d0_spectroscopic.png`（分光学的解離エネルギー D0（0 K；高精度分光）：結合エネルギーの独立ベースライン（Step 7.12））
- `output/quantum/molecular_dissociation_d0_spectroscopic_metrics.json`（数値）
- `output/quantum/molecular_transitions_exomol_baseline.png`（分子遷移（一次線リスト；H2/HD: ExoMol / D2: MOLAT）：代表遷移（Einstein A top10）を固定（Step 7.12））
- `output/quantum/molecular_transitions_exomol_baseline_metrics.json`（数値）
- `output/quantum/molecular_transitions_exomol_baseline_selected.csv`（選別された遷移一覧）

---

## 物性（凝縮系）

- `output/quantum/condensed_silicon_lattice_baseline.png`（Si格子定数（NIST CODATA）：基準値（ターゲット）固定（Step 7.14.1））
- `output/quantum/condensed_silicon_lattice_baseline_metrics.json`（数値）
- `output/quantum/condensed_silicon_lattice_baseline.csv`（一覧）
- `output/quantum/condensed_silicon_heat_capacity_baseline.png`（Si比熱 Cp(T)（NIST WebBook condensed；Shomate）：基準曲線固定（Step 7.14.2））
- `output/quantum/condensed_silicon_heat_capacity_baseline_metrics.json`（数値）
- `output/quantum/condensed_silicon_heat_capacity_baseline.csv`（一覧）
- `output/quantum/condensed_silicon_heat_capacity_debye_baseline.png`（Si低温比熱 Cp(T)（NIST-JANAF）＋Debye fit（θ_D）：圧縮ターゲット固定（Step 7.14.3））
- `output/quantum/condensed_silicon_heat_capacity_debye_baseline_metrics.json`（数値）
- `output/quantum/condensed_silicon_heat_capacity_debye_baseline.csv`（一覧）
- `output/quantum/condensed_silicon_resistivity_temperature_coefficient_baseline.png`（Si抵抗率 ρ(T)（NBS IR 74-496 Appendix E）＋室温近傍の温度係数：基準値固定（Step 7.14.4））
- `output/quantum/condensed_silicon_resistivity_temperature_coefficient_baseline_metrics.json`（数値）
- `output/quantum/condensed_silicon_resistivity_temperature_coefficient_baseline.csv`（一覧）
- `output/quantum/condensed_silicon_thermal_expansion_baseline.png`（Si熱膨張係数 α(T)（NIST TRC cryogenics）：基準曲線固定（Step 7.14.5））
- `output/quantum/condensed_silicon_thermal_expansion_baseline_metrics.json`（数値）
- `output/quantum/condensed_silicon_thermal_expansion_baseline.csv`（一覧）
- `output/quantum/condensed_silicon_thermal_expansion_gruneisen_minimal_model.png`（Si熱膨張係数：Debye–Grüneisen 最小モデル（α≈A·Cv）の成立性チェック（Step 7.14.8））
- `output/quantum/condensed_silicon_thermal_expansion_gruneisen_minimal_model_metrics.json`（数値）
- `output/quantum/condensed_silicon_thermal_expansion_gruneisen_minimal_model.csv`（一覧）
- `output/quantum/condensed_silicon_thermal_expansion_gruneisen_gammaT_model.png`（Si熱膨張係数：Debye–Grüneisen 最小拡張（A_eff(T)=A_inf·tanh((T−T0)/ΔT)）の成立性チェック（Step 7.14.9））
- `output/quantum/condensed_silicon_thermal_expansion_gruneisen_gammaT_model_metrics.json`（数値）
- `output/quantum/condensed_silicon_thermal_expansion_gruneisen_gammaT_model.csv`（一覧）
- `output/quantum/condensed_silicon_thermal_expansion_gruneisen_two_branch_model.png`（Si熱膨張係数：Debye–Grüneisen 2枝モデル（α≈A1·Cv(θ1)+A2·Cv(θ2)）の成立性チェック（Step 7.14.10））
- `output/quantum/condensed_silicon_thermal_expansion_gruneisen_two_branch_model_metrics.json`（数値）
- `output/quantum/condensed_silicon_thermal_expansion_gruneisen_two_branch_model.csv`（一覧）
- `output/quantum/condensed_silicon_thermal_expansion_gruneisen_debye_einstein_model.png`（Si熱膨張係数：Debye+Einstein 混合モデル（α≈A_D·Cv_D(θ_D)+A_E·Cv_E(θ_E)）の成立性チェック（Step 7.14.11））
- `output/quantum/condensed_silicon_thermal_expansion_gruneisen_debye_einstein_model_metrics.json`（数値）
- `output/quantum/condensed_silicon_thermal_expansion_gruneisen_debye_einstein_model.csv`（一覧）
- `output/quantum/condensed_silicon_thermal_expansion_gruneisen_polyA_model.png`（Si熱膨張係数：A_eff(T) 多項式 ansatz（α≈A_eff(T)·Cv_Debye）の診断（Step 7.14.12））
- `output/quantum/condensed_silicon_thermal_expansion_gruneisen_polyA_model_metrics.json`（数値）
- `output/quantum/condensed_silicon_thermal_expansion_gruneisen_polyA_model.csv`（一覧）
- `output/quantum/condensed_silicon_thermal_expansion_gruneisen_cp_proxy_gammaT_model.png`（Si熱膨張係数：JANAF Cp を proxy に取り込んだ Grüneisen ansatz（α≈A_inf·tanh((T−T0)/ΔT)·Cp_proxy）の診断（Step 7.14.13））
- `output/quantum/condensed_silicon_thermal_expansion_gruneisen_cp_proxy_gammaT_model_metrics.json`（数値）
- `output/quantum/condensed_silicon_thermal_expansion_gruneisen_cp_proxy_gammaT_model.csv`（一覧）
- `output/quantum/condensed_silicon_bulk_modulus_baseline.png`（Siバルク弾性率 B(T) の参照曲線（Ioffe 弾性定数；400K以上は線形近似、未満は一定；Step 7.14.16））
- `output/quantum/condensed_silicon_bulk_modulus_baseline_metrics.json`（数値）
- `output/quantum/condensed_silicon_bulk_modulus_baseline.csv`（一覧）
- `output/quantum/condensed_silicon_thermal_expansion_gruneisen_cp_proxy_gammaT_bulkmodulus_model.png`（Si熱膨張係数：Cp_proxy + B(T) を用いた γ(T) tanh ansatz の診断（α≈γ·Cp_proxy/(B·V_m)）（Step 7.14.17））
- `output/quantum/condensed_silicon_thermal_expansion_gruneisen_cp_proxy_gammaT_bulkmodulus_model_metrics.json`（数値）
- `output/quantum/condensed_silicon_thermal_expansion_gruneisen_cp_proxy_gammaT_bulkmodulus_model.csv`（一覧）
- `output/quantum/condensed_silicon_thermal_expansion_gruneisen_debye_einstein_two_branch_model.png`（Si熱膨張係数：Debye+Einstein×2 混合モデル（α≈A_D·Cv_D(θ_D)+A_E1·Cv_E(θ_E1)+A_E2·Cv_E(θ_E2)）の成立性チェック（Step 7.14.14））
- `output/quantum/condensed_silicon_thermal_expansion_gruneisen_debye_einstein_two_branch_model_metrics.json`（数値）
- `output/quantum/condensed_silicon_thermal_expansion_gruneisen_debye_einstein_two_branch_model.csv`（一覧）
- `output/quantum/condensed_silicon_thermal_expansion_gruneisen_holdout_splits.png`（Si熱膨張係数：temperature-split holdout（train→test）で ansatz の手続き依存性を定量化（Step 7.14.15））
- `output/quantum/condensed_silicon_thermal_expansion_gruneisen_holdout_splits_metrics.json`（数値）
- `output/quantum/condensed_silicon_thermal_expansion_gruneisen_holdout_splits.csv`（一覧）
- `output/quantum/condensed_silicon_thermal_expansion_gruneisen_ioffe_phonon_anchors_model.png`（Si熱膨張係数：Ioffe の phonon 周波数（代表値）を用いて θ_E を離散アンカーとして固定した Debye+Einstein×2 モデル（Step 7.14.18））
- `output/quantum/condensed_silicon_thermal_expansion_gruneisen_ioffe_phonon_anchors_model_metrics.json`（数値）
- `output/quantum/condensed_silicon_thermal_expansion_gruneisen_ioffe_phonon_anchors_model.csv`（一覧）
- `output/quantum/condensed_silicon_thermal_expansion_gruneisen_ioffe_phonon_anchors_three_einstein_model.png`（Si熱膨張係数：Ioffe の phonon 周波数（代表値）を 3枝（Einstein×3）へ拡張して固定したモデル（Step 7.14.19））
- `output/quantum/condensed_silicon_thermal_expansion_gruneisen_ioffe_phonon_anchors_three_einstein_model_metrics.json`（数値）
- `output/quantum/condensed_silicon_thermal_expansion_gruneisen_ioffe_phonon_anchors_three_einstein_model.csv`（一覧）
- `output/quantum/condensed_silicon_thermal_expansion_gruneisen_phonon_dos_mode_gamma_model.png`（Si熱膨張係数：phonon DOS（proxy）で mode-weighting を凍結した 2-group（acoustic/optical）Grüneisen モデル（Step 7.14.20））
- `output/quantum/condensed_silicon_thermal_expansion_gruneisen_phonon_dos_mode_gamma_model_metrics.json`（数値）
- `output/quantum/condensed_silicon_thermal_expansion_gruneisen_phonon_dos_mode_gamma_model.csv`（一覧）
- `output/quantum/condensed_silicon_thermal_expansion_gruneisen_phonon_dos_mode_gamma_three_group_model.png`（Si熱膨張係数：phonon DOS（proxy）で mode-weighting を凍結した 3-group（TA/LA/optical）Grüneisen モデル（Step 7.14.20））
- `output/quantum/condensed_silicon_thermal_expansion_gruneisen_phonon_dos_mode_gamma_three_group_model_metrics.json`（数値）
- `output/quantum/condensed_silicon_thermal_expansion_gruneisen_phonon_dos_mode_gamma_three_group_model.csv`（一覧）
- `output/quantum/condensed_silicon_thermal_expansion_gruneisen_phonon_dos_mode_gamma_four_group_model.png`（Si熱膨張係数：phonon DOS（proxy）で mode-weighting を凍結した 4-group（TA/LA/TO/LO）Grüneisen モデル（Step 7.14.20））
- `output/quantum/condensed_silicon_thermal_expansion_gruneisen_phonon_dos_mode_gamma_four_group_model_metrics.json`（数値）
- `output/quantum/condensed_silicon_thermal_expansion_gruneisen_phonon_dos_mode_gamma_four_group_model.csv`（一覧）
- `output/quantum/condensed_silicon_thermal_expansion_gruneisen_phonon_dos_mode_gamma_optical_softening_linear_model.png`（Si熱膨張係数：phonon DOS split（2-group）に optical softening（線形；fをgrid scan）を加えた検査（Step 7.14.20））
- `output/quantum/condensed_silicon_thermal_expansion_gruneisen_phonon_dos_mode_gamma_optical_softening_linear_model_metrics.json`（数値）
- `output/quantum/condensed_silicon_thermal_expansion_gruneisen_phonon_dos_mode_gamma_optical_softening_linear_model.csv`（一覧）
- `output/quantum/condensed_silicon_thermal_expansion_gruneisen_phonon_dos_mode_gamma_three_group_optical_softening_linear_model.png`（Si熱膨張係数：phonon DOS split（3-group）に optical softening（線形；fをgrid scan）を加えた検査（Step 7.14.20））
- `output/quantum/condensed_silicon_thermal_expansion_gruneisen_phonon_dos_mode_gamma_three_group_optical_softening_linear_model_metrics.json`（数値）
- `output/quantum/condensed_silicon_thermal_expansion_gruneisen_phonon_dos_mode_gamma_three_group_optical_softening_linear_model.csv`（一覧）
- `output/quantum/condensed_silicon_thermal_expansion_gruneisen_phonon_dos_mode_gamma_four_group_optical_softening_linear_model.png`（Si熱膨張係数：phonon DOS split（4-group）に optical softening（線形；fをgrid scan）を加えた検査（Step 7.14.20））
- `output/quantum/condensed_silicon_thermal_expansion_gruneisen_phonon_dos_mode_gamma_four_group_optical_softening_linear_model_metrics.json`（数値）
- `output/quantum/condensed_silicon_thermal_expansion_gruneisen_phonon_dos_mode_gamma_four_group_optical_softening_linear_model.csv`（一覧）
- `output/quantum/condensed_silicon_thermal_expansion_gruneisen_phonon_dos_mode_gamma_optical_softening_raman_shape_model.png`（Si熱膨張係数：phonon DOS split（2-group）に Raman ω(T) 由来の softening 形状を加えた検査（Step 7.14.20））
- `output/quantum/condensed_silicon_thermal_expansion_gruneisen_phonon_dos_mode_gamma_optical_softening_raman_shape_model_metrics.json`（数値）
- `output/quantum/condensed_silicon_thermal_expansion_gruneisen_phonon_dos_mode_gamma_optical_softening_raman_shape_model.csv`（一覧）
- `output/quantum/condensed_silicon_thermal_expansion_gruneisen_phonon_dos_mode_gamma_three_group_optical_softening_raman_shape_model.png`（Si熱膨張係数：phonon DOS split（3-group）に Raman ω(T) 由来の softening 形状を加えた検査（Step 7.14.20））
- `output/quantum/condensed_silicon_thermal_expansion_gruneisen_phonon_dos_mode_gamma_three_group_optical_softening_raman_shape_model_metrics.json`（数値）
- `output/quantum/condensed_silicon_thermal_expansion_gruneisen_phonon_dos_mode_gamma_three_group_optical_softening_raman_shape_model.csv`（一覧）
- `output/quantum/condensed_silicon_thermal_expansion_gruneisen_phonon_dos_mode_gamma_four_group_optical_softening_raman_shape_model.png`（Si熱膨張係数：phonon DOS split（4-group）に Raman ω(T) 由来の softening 形状を加えた検査（Step 7.14.20））
- `output/quantum/condensed_silicon_thermal_expansion_gruneisen_phonon_dos_mode_gamma_four_group_optical_softening_raman_shape_model_metrics.json`（数値）
- `output/quantum/condensed_silicon_thermal_expansion_gruneisen_phonon_dos_mode_gamma_four_group_optical_softening_raman_shape_model.csv`（一覧）
- `output/quantum/condensed_silicon_thermal_expansion_gruneisen_phonon_dos_mode_gamma_dos_softening_kim2015_linear_model.png`（Si熱膨張係数：Kim2015（INS）の mean shift から凍結した global ω(T) softening（線形proxy）を 2-group DOS拘束へ適用した検査（Step 7.14.20））
- `output/quantum/condensed_silicon_thermal_expansion_gruneisen_phonon_dos_mode_gamma_dos_softening_kim2015_linear_model_metrics.json`（数値）
- `output/quantum/condensed_silicon_thermal_expansion_gruneisen_phonon_dos_mode_gamma_dos_softening_kim2015_linear_model.csv`（一覧）
- `output/quantum/condensed_silicon_thermal_expansion_gruneisen_phonon_dos_mode_gamma_three_group_dos_softening_kim2015_linear_model.png`（Si熱膨張係数：Kim2015（INS）の mean shift から凍結した global ω(T) softening（線形proxy）を 3-group DOS拘束へ適用した検査（Step 7.14.20））
- `output/quantum/condensed_silicon_thermal_expansion_gruneisen_phonon_dos_mode_gamma_three_group_dos_softening_kim2015_linear_model_metrics.json`（数値）
- `output/quantum/condensed_silicon_thermal_expansion_gruneisen_phonon_dos_mode_gamma_three_group_dos_softening_kim2015_linear_model.csv`（一覧）
- `output/quantum/condensed_silicon_thermal_expansion_gruneisen_phonon_dos_mode_gamma_four_group_dos_softening_kim2015_linear_model.png`（Si熱膨張係数：Kim2015（INS）の mean shift から凍結した global ω(T) softening（線形proxy）を 4-group DOS拘束へ適用した検査（Step 7.14.20））
- `output/quantum/condensed_silicon_thermal_expansion_gruneisen_phonon_dos_mode_gamma_four_group_dos_softening_kim2015_linear_model_metrics.json`（数値）
- `output/quantum/condensed_silicon_thermal_expansion_gruneisen_phonon_dos_mode_gamma_four_group_dos_softening_kim2015_linear_model.csv`（一覧）
- `output/quantum/condensed_ofhc_copper_thermal_conductivity_baseline.png`（OFHC copper 熱伝導率 κ(T)（NIST TRC cryogenics；RRR依存）：基準曲線固定（Step 7.14.6））
- `output/quantum/condensed_ofhc_copper_thermal_conductivity_baseline_metrics.json`（数値）
- `output/quantum/condensed_ofhc_copper_thermal_conductivity_baseline.csv`（一覧）

---

## 統計力学・熱力学

- `output/quantum/thermo_blackbody_radiation_baseline.png`（黒体放射：u∝T^4 と nγ∝T^3 の基準スケールを固定（Step 7.15.1））
- `output/quantum/thermo_blackbody_radiation_baseline_metrics.json`（数値）
- `output/quantum/thermo_blackbody_radiation_baseline.csv`（一覧）
- `output/quantum/thermo_blackbody_entropy_baseline.png`（黒体：エントロピー密度 s∝T^3 と s/(nγ k_B)=const を固定（Step 7.15.2））
- `output/quantum/thermo_blackbody_entropy_baseline_metrics.json`（数値）
- `output/quantum/thermo_blackbody_entropy_baseline.csv`（一覧）

---

## 量子（ベルテスト）

- `output/quantum/nist_belltest_time_tag_bias.png`（NIST belltestdata: time-tag の設定依存と window 掃引）
- `output/quantum/nist_belltest_time_tag_bias__03_43_afterfixingModeLocking_s3600.png`（NIST: 03_43 run（先頭3600秒）の再解析）
- `output/quantum/nist_belltest_time_tag_bias_metrics.json`（数値）
- `output/quantum/nist_belltest_coincidence_sweep.csv`（window sweep）
- `output/quantum/nist_belltest_trial_based__03_43_afterfixingModeLocking_s3600.png`（NIST: trial-based vs coincidence-based（03_43 run））
- `output/quantum/delft_hensen2015_chsh.png`（Delft/Hensen 2015: CHSH S と event-ready window start 感度）
- `output/quantum/delft_hensen2016_srep30289_chsh.png`（Delft/Hensen 2016（Sci Rep 30289）: CHSH（combined）と start offset 感度）
- `output/quantum/delft_hensen2015_chsh_metrics.json`（数値）
- `output/quantum/delft_hensen2015_chsh_sweep_start_offset.csv`（start offset sweep）
- `output/quantum/weihs1998_chsh_sweep__weihs1998_longdist_longdist1.png`（Weihs 1998: time-tag の coincidence window 掃引と CHSH |S| の感度）
- `output/quantum/weihs1998_chsh_sweep_summary__multi_subdirs.png`（Weihs 1998: 複数条件（subdir）での sweep まとめ）
- `output/quantum/weihs1998_chsh_sweep_metrics__weihs1998_longdist_longdist1.json`（数値）
- `output/quantum/weihs1998_chsh_sweep__weihs1998_longdist_longdist1.csv`（window sweep）
- `output/quantum/bell_selection_sensitivity_summary.png`（横断まとめ：selection感度（window/offset）＋自然窓推奨線（dtピークの背景差し引き＝accidental proxy / trial照合））
- `output/quantum/bell/falsification_pack.png`（横断まとめ：Bell反証条件パック（selection ratio と delay（Δmedian; z）を運用閾値つきで固定））
- `output/quantum/bell/longterm_consistency.png`（横断まとめ：Bell入口指標（R_sel と z_delay）の長期一貫性（cross-dataset）と natural window freeze の固定）
- `output/quantum/bell/systematics_decomposition_15items.png`（横断まとめ：Bell系統誤差の 15項目分解（median budget）と dataset別総和、および項目間相関（ヒートマップ））
- `output/quantum/bell/cross_dataset_covariance.png`（横断まとめ：cross-dataset の sweep profile を共通グリッドへ射影し、共分散（相関）構造と固有モード要約を固定）

---

## 量子（重力×量子干渉）

- `output/quantum/cow_phase_shift.png`（COW（中性子干渉）：位相差のスケーリング（一次ソース））
- `output/quantum/cow_phase_shift_metrics.json`（数値）
- `output/quantum/atom_interferometer_gravimeter_phase.png`（原子干渉計重力計：位相差 φ≈k_eff g T² のスケーリング（一次ソース））
- `output/quantum/atom_interferometer_gravimeter_phase_metrics.json`（数値）
- `output/quantum/optical_clock_chronometric_leveling.png`（光格子時計（87Sr）：chronometric leveling（ΔU from clock vs geodesy））
- `output/quantum/optical_clock_chronometric_leveling_metrics.json`（数値）
- `output/quantum/gravity_quantum_interference_delta_predictions.png`（P-model vs GR（時計写像）の差分スケール：必要精度（3σ）と代表精度の比較（Earth/Sun/強場の例を JSON に固定））
- `output/quantum/gravity_quantum_interference_delta_predictions.json`（数値＋仮定列）

---

## 量子（物質波干渉／de Broglie）

- `output/quantum/electron_double_slit_interference.png`（電子二重スリット：P12 と P1+P2（干渉項）の再現（一次ソース））
- `output/quantum/electron_double_slit_interference_metrics.json`（数値）
- `output/quantum/de_broglie_precision_alpha_consistency.png`（de Broglie 精密：α（原子反跳）と α（電子 g-2）の整合チェック）
- `output/quantum/de_broglie_precision_alpha_consistency_metrics.json`（数値）
- `output/quantum/matter_wave_interference_precision_audit.png`（物質波干渉の横断監査：電子（縞スケール）・原子（α整合/精度ギャップ）・分子（同位体スケーリング）を同一I/Fで固定（Step 7.16.13））
- `output/quantum/matter_wave_interference_precision_audit_metrics.json`（数値）

---

## 量子（重力誘起デコヒーレンス）

- `output/quantum/gravity_induced_decoherence.png`（重力誘起デコヒーレンス：観測量（V）と、P-model側の雑音パラメータ（σy）の必要条件）
- `output/quantum/gravity_induced_decoherence_metrics.json`（数値）

---

## 量子（光の量子干渉）

- `output/quantum/photon_quantum_interference.png`（光の量子干渉：単一光子visibility、HOM visibility、スクイーズド光（dB/損失下限）と低周波ノイズPSD）
- `output/quantum/photon_quantum_interference_metrics.json`（数値）
- `output/quantum/hom_squeezed_light_unified_audit.png`（HOMとスクイーズド光の統合監査：古典閾値有意性、遅延依存、分散比、PSD比を同一I/Fで固定（Step 7.16.14））
- `output/quantum/hom_squeezed_light_unified_audit_metrics.json`（数値）

---

## 量子（真空・QED精密）

- `output/quantum/qed_vacuum_precision.png`（真空・QED精密：Casimir 力スケール、Lamb shift のスケーリング（Z^4/Z^6）、核サイズ寄与の例）
- `output/quantum/qed_vacuum_precision_metrics.json`（数値）

---

## 量子（原子核）

- `output/quantum/nuclear_binding_deuteron.png`（原子核ベースライン：deuteron の束縛エネルギーBとサイズ拘束（r_d, 1/κ））
- `output/quantum/nuclear_binding_light_nuclei.png`（軽核ベースライン：A=2,3,4 の束縛エネルギー（質量欠損）と、利用可能な電荷半径（p/d/α）を一次固定（Step 7.13.9））
- `output/quantum/nuclear_binding_representative_nuclei.png`（代表核種ベースライン：軽核＋代表安定核の B/B/A（AME2020）と電荷半径（IAEA）を統合して一次固定（Step 7.13.11））
- `output/quantum/nuclear_a_dependence_mean_field.png`（A依存チェック：7.13.8 の固定 u-profile を用いた uniform mean-field による B/A の整合性チェック（Step 7.13.12））
- `output/quantum/nuclear_a_dependence_mean_field_density_repulsion.png`（A依存チェック：2成分Fermi＋Coulomb に加え、最小の多体反発 proxy（`E_rep/A=Cρ^2`；1点凍結）を導入した切り分け（Step 7.13.13））
- `output/quantum/nuclear_a_dependence_mean_field_repulsion_radii_frozen.png`（A依存チェック：多体反発 proxy（`E_rep/A=Cρ^2`）の C を“電荷半径の平衡条件（dE/dR=0）”で凍結した切り分け（Step 7.13.14））
- `output/quantum/nuclear_a_dependence_hf_three_body_radii_frozen.png`（A依存チェック：Hartree–Fock（pp/nn の exchange/Fock）＋3体項 proxy（`E_3/A=C3ρ^2`）で切り分け（C3は電荷半径の平衡条件で凍結；Step 7.13.15））
- `output/quantum/nuclear_a_dependence_hf_three_body_uncertainty.png`（不確かさ分解：核種ごとの一次σ（AME2020 B/A σ、IAEA 電荷半径 σ）と、重核での `C3_est` 散らばり（MAD→σ）から `σ_total` を構成（Step 7.13.15.2））
- `output/quantum/nuclear_a_dependence_hf_three_body_uncertainty_set_systematic.png`（系統追加：eq18/eq19 の予測差を `σ_set=0.5|eq18-eq19|` として `σ_total` に加え、棄却用の不確かさ予算を強化（Step 7.13.15.3））
- `output/quantum/nuclear_a_dependence_hf_three_body_uncertainty_radii_source_systematic.png`（系統追加：CODATA（NIST Cuu）と IAEA radii の差を `σ_r_source=0.5|CODATA-IAEA|` として半径系統へ上乗せし、半径ソース選択依存を明示（Step 7.13.15.4））
- `output/quantum/nuclear_a_dependence_hf_three_body_rejection_budget.png`（総合棄却予算：`σ_final^2=σ_total^2+σ_set^2+σ_r_source^2` を固定し、`z=residual/σ_final` と適用範囲（A>=16）を明示（Step 7.13.15.5））
- `output/quantum/nuclear_a_dependence_hf_three_body_extended_dataset.png`（データ数拡張：AME2020×IAEA radii を大規模 join（N=955）し、残差を shell（magic Z/N）・surface（A-bin）で切り分けて固定出力化（Step 7.13.15.6））
- `output/quantum/nuclear_a_dependence_hf_three_body_surface_term.png`（surface項の明示：`C3_est_from_radii ≈ C3∞ + C_surf*x`（x=1/(6ρ^2R)）を凍結し、`C3_eff` による予測と z を更新（Step 7.13.15.7））
- `output/quantum/nuclear_a_dependence_hf_three_body_shell_term.png`（shell系統の明示：magic Z/N を category とし、`σ_final_shell^2 = σ_final^2 + σ_shell^2` で棄却予算を更新（Step 7.13.15.8））
- `output/quantum/nuclear_a_dependence_hf_three_body_separation_energies.png`（決着点の固定：B/Aではなく separation energies（S_n/S_2n）と shell-gap を固定出力化し、shell を「補正（deterministic）」へ昇格するかの判断材料を作成（Step 7.13.15.9））
- `output/quantum/nuclear_a_dependence_hf_three_body_shell_correction.png`（deterministic shell correction：最小2自由度（magic指標）の補正を Sn（magic N=50/82）で凍結し、S_n/S_2n と shell-gap で外挿評価（Step 7.13.15.10））
- `output/quantum/nuclear_a_dependence_hf_three_body_shell_quantization.png`（殻量子化写像：殻内占有分率 `S_shell(N/Z)` と `ħω(A)` による最小補正（κのみ）を導入し、shell-gap の外挿性能を検証（Step 7.13.15.11））
- `output/quantum/nuclear_a_dependence_hf_three_body_shell_quantization_asym.png`（殻量子化写像（N/Z非対称）：`kN,kZ` を導入し、kN を `gap_n`、kZ を `gap_p` で別々に凍結して外挿評価（Step 7.13.15.12））
- `output/quantum/nuclear_a_dependence_hf_three_body_shell_degeneracy_model.png`（殻境界の内部生成：Nilsson-like 最小スペクトルから殻容量（MODEL_MAGIC）を生成し、同一の凍結手順で 1n/1p shell-gap の外挿改善を検証（Step 7.13.15.13））
- `output/quantum/nuclear_a_dependence_hf_three_body_shell_degeneracy_pairing_model.png`（最小ペアリング項（OES 3-point 凍結）を追加し、1n/1p shell-gap の外挿改善を検証（Step 7.13.15.14））
- `output/quantum/nuclear_a_dependence_hf_three_body_shell_degeneracy_pairing_refreeze.png`（pairing を固定した上で kN/kZ を再凍結し、1n/1p shell-gap の外挿を再評価（Step 7.13.15.15））
- `output/quantum/nuclear_a_dependence_hf_three_body_shell_degeneracy_pairing_sep_refreeze.png`（pairing の最小形を改善（odd-A を含む/中性子・陽子分離）し、kN/kZ 再凍結の下で外挿判定（Step 7.13.15.16））
- `output/quantum/nuclear_a_dependence_hf_three_body_shell_degeneracy_collective_pn.png`（本質的相関（pn-collective proxy）を最小自由度で導入し、shell-gap 外挿で判定（Step 7.13.15.17））
- `output/quantum/nuclear_a_dependence_hf_three_body_shell_degeneracy_collective_q_sep.png`（識別性を保った最小増分（2DoF）として `q(N),q(Z)` を導入し、shell-gap 外挿で判定（Step 7.13.15.18））
- `output/quantum/nuclear_a_dependence_hf_three_body_shellgap_no_go_pack.png`（no-go pack：平均場+殻補正+最小相関では other-magic の 1n/1p shell-gap 外挿が救えないことを機械的に確定（Step 7.13.15.19））
- `output/quantum/nuclear_a_dependence_hf_three_body_shellgap_failure_modes.png`（failure modes：shell-gap 外挿の失敗を magic と A で分解し、次の設計点（domain/coverage）を仕様化（Step 7.13.15.20））
- `output/quantum/nuclear_a_dependence_hf_three_body_shellgap_coverage_expanded.png`（coverage expanded：凍結した半径モデルで AME2020 mass table へ外挿し、shell-gap 決着点（Z=28等）を増やす（Step 7.13.15.21））
- `output/quantum/nuclear_a_dependence_hf_three_body_shellgap_decision_expanded.png`（decision scan：domain A_min×shell 写像2候補をスキャンし、expanded set の strict 判定で no-go を再評価（Step 7.13.15.22））
- `output/quantum/nuclear_a_dependence_hf_three_body_shellgap_decision_expanded_radii_isospin.png`（decision scan：独立拘束DoF（半径 isospin 拡張；radii-only 凍結）を導入し、expanded set の strict 判定で再評価（Step 7.13.15.23））
- `output/quantum/nuclear_a_dependence_hf_three_body_shellgap_decision_expanded_coulomb_exchange.png`（decision scan：Coulomb exchange（Slater）を自由度なしで追加し、expanded set の strict 判定で再評価（Step 7.13.15.24））
- `output/quantum/nuclear_a_dependence_hf_three_body_shellgap_decision_expanded_coulomb_finite_size.png`（decision scan：Coulomb/finite-size（2pF+Slater）を自由度なしで追加し、expanded set の strict 判定で再評価（Step 7.13.15.25））
- `output/quantum/nuclear_a_dependence_hf_three_body_shellgap_decision_expanded_radii_isospin_coulomb_finite_size.png`（decision scan：半径 isospin 拡張（独立凍結）＋ Coulomb/finite-size（2pF+Slater）を組み合わせ、expanded set の strict 判定で再評価（Step 7.13.15.26））
- `output/quantum/nuclear_a_dependence_hf_three_body_shellgap_decision_expanded_shell_i_dep.png`（decision scan：shell refreeze を isospin 依存で拡張し（train 凍結→other 棄却）、expanded set の strict 判定で再評価（Step 7.13.15.27））
- `output/quantum/nuclear_a_dependence_hf_three_body_shellgap_decision_expanded_shell_i_dep_beta2.png`（decision scan：独立拘束の変形入力 β2（NNDC B(E2) adopted）を Coulomb shape factor として追加し、expanded set の strict 判定で再評価（Step 7.13.15.28））
- `output/quantum/nuclear_a_dependence_hf_three_body_shellgap_decision_expanded_shell_i_dep_beta2_surface_cov.png`（decision scan：β2 を Coulomb+surface に deterministic に反映し、coverage（direct-only vs imputed）を strict に統合して再評価（Step 7.13.15.29））
- `output/quantum/nuclear_a_dependence_hf_three_body_shellgap_decision_expanded_shell_i_dep_beta2_surface_cov_e2plus.png`（decision scan：分光一次ソース E(2+_1)（NNDC；transitionEnergy）を追加し、分光ピーク（±2近傍）strict を統合して再評価（Step 7.13.15.30））
- `output/quantum/nuclear_a_dependence_hf_three_body_e2plus_diagnostics_expanded_shell_i_dep_beta2_surface_cov_e2plus.png`（E(2+_1) 診断：even-even の E(2+_1) を N/Z で可視化し、局所ピーク（±2近傍）を shell closure proxy として固定（Step 7.13.15.30））
- `output/quantum/nuclear_a_dependence_hf_three_body_shellgap_decision_expanded_shell_i_dep_beta2_surface_cov_e2plus_nudat3.png`（decision scan：分光一次ソース E(2+_1)（NuDat 3.0；firstTwoPlusEnergy）で coverage を拡大し、同一の分光ピーク（±2近傍）strict を統合して再評価（Step 7.13.15.31））
- `output/quantum/nuclear_a_dependence_hf_three_body_e2plus_diagnostics_expanded_shell_i_dep_beta2_surface_cov_e2plus_nudat3.png`（E(2+_1) 診断：NuDat 3.0 の E(2+_1)（even-even）を N/Z で可視化し、局所ピーク（±2近傍）を shell closure proxy として固定（Step 7.13.15.31））
- `output/quantum/nuclear_a_dependence_hf_three_body_shellgap_decision_expanded_shell_i_dep_beta2_surface_cov_spectro_multi_nudat3.png`（decision scan：NuDat 3.0 の分光指標（E(2+_1), E(4+_1), E(3-_1), R4/2）を local extrema で strict に統合し、expanded set の判定を再評価（Step 7.13.15.32））
- `output/quantum/nuclear_a_dependence_hf_three_body_spectroscopy_diagnostics_expanded_shell_i_dep_beta2_surface_cov_spectro_multi_nudat3.png`（分光診断：NuDat 3.0 の multi 指標を N/Z で可視化し、local extrema を shell closure proxy として固定（Step 7.13.15.32））
- `output/quantum/nuclear_a_dependence_hf_three_body_shellgap_decision_expanded_shell_i_dep_beta2_surface_cov_spectro_multi_nudat3_shellS_g2.png`（decision scan：S_shell の g^power 正規化を g^2（power=2）へ変更し、expanded set の strict 判定で再評価（Step 7.13.15.33））
- `output/quantum/nuclear_a_dependence_hf_three_body_spectroscopy_diagnostics_expanded_shell_i_dep_beta2_surface_cov_spectro_multi_nudat3_shellS_g2.png`（分光診断：同上（shellS_g2）で N/Z 診断（Step 7.13.15.33））
- `output/quantum/nuclear_a_dependence_hf_three_body_shellgap_decision_expanded_shell_i_dep_beta2_surface_cov_spectro_multi_nudat3_radii_kink.png`（decision scan：電荷半径の kink（同位体シフト）を shell closure proxy として strict にAND統合し、expanded set の判定を再評価（Step 7.13.15.34））
- `output/quantum/nuclear_a_dependence_hf_three_body_spectroscopy_diagnostics_expanded_shell_i_dep_beta2_surface_cov_spectro_multi_nudat3_radii_kink.png`（分光診断：同上（radii_kink）で N/Z 診断（Step 7.13.15.34））
- `output/quantum/nuclear_a_dependence_hf_three_body_radii_kink_diagnostics_expanded_shell_i_dep_beta2_surface_cov_spectro_multi_nudat3_radii_kink.png`（半径 kink 診断：IAEA radii の Δ²r（step=2）を N/Z で可視化し、kink を shell closure proxy として固定（Step 7.13.15.34））
- `output/quantum/nuclear_a_dependence_hf_three_body_shellgap_decision_expanded_shell_i_dep_beta2_surface_cov_spectro_multi_nudat3_radii_kink_delta2r.png`（decision scan：電荷半径 kink を「観測量（Δ²r）」として Δ²r_pred vs Δ²r_obs を直接比較し、正規化残差（σ_obs）を strict にAND統合して再評価（Step 7.13.15.35））
- `output/quantum/nuclear_a_dependence_hf_three_body_spectroscopy_diagnostics_expanded_shell_i_dep_beta2_surface_cov_spectro_multi_nudat3_radii_kink_delta2r.png`（分光診断：同上（radii_kink_delta2r）で N/Z 診断（Step 7.13.15.35））
- `output/quantum/nuclear_a_dependence_hf_three_body_radii_kink_delta2r_diagnostics_expanded_shell_i_dep_beta2_surface_cov_spectro_multi_nudat3_radii_kink_delta2r.png`（半径 kink Δ²r 残差診断：|Δ²r_pred-Δ²r_obs|/σ_obs を N/Z で可視化（Step 7.13.15.35））
- `output/quantum/nuclear_a_dependence_hf_three_body_shellgap_decision_expanded_shell_i_dep_beta2_surface_cov_spectro_multi_nudat3_radii_kink_delta2r_radius_shell.png`（decision scan：半径写像に最小 shell 項（自由度1：r_shell）を追加し、Δ²r strict を同一閾値で再評価（Step 7.13.15.36））
- `output/quantum/nuclear_a_dependence_hf_three_body_spectroscopy_diagnostics_expanded_shell_i_dep_beta2_surface_cov_spectro_multi_nudat3_radii_kink_delta2r_radius_shell.png`（分光診断：同上（radii_kink_delta2r_radius_shell）で N/Z 診断（Step 7.13.15.36））
- `output/quantum/nuclear_a_dependence_hf_three_body_radii_kink_delta2r_diagnostics_expanded_shell_i_dep_beta2_surface_cov_spectro_multi_nudat3_radii_kink_delta2r_radius_shell.png`（半径 kink Δ²r 残差診断：|Δ²r_pred-Δ²r_obs|/σ_obs を N/Z で可視化（Step 7.13.15.36））
- `output/quantum/nuclear_a_dependence_hf_three_body_shellgap_decision_expanded_shell_i_dep_beta2_surface_cov_spectro_multi_nudat3_radii_kink_delta2r_radius_shell_nz.png`（decision scan：半径写像を N/Z 分離（自由度2：r_shell_N, r_shell_Z）で拡張し、Δ²r strict を同一閾値で再評価（Step 7.13.15.37））
- `output/quantum/nuclear_a_dependence_hf_three_body_spectroscopy_diagnostics_expanded_shell_i_dep_beta2_surface_cov_spectro_multi_nudat3_radii_kink_delta2r_radius_shell_nz.png`（分光診断：同上（radii_kink_delta2r_radius_shell_nz）で N/Z 診断（Step 7.13.15.37））
- `output/quantum/nuclear_a_dependence_hf_three_body_radii_kink_delta2r_diagnostics_expanded_shell_i_dep_beta2_surface_cov_spectro_multi_nudat3_radii_kink_delta2r_radius_shell_nz.png`（半径 kink Δ²r 残差診断：|Δ²r_pred-Δ²r_obs|/σ_obs を N/Z で可視化（Step 7.13.15.37））
- `output/quantum/nuclear_np_scattering_baseline.png`（np散乱ベースライン：低エネルギーパラメータ（a_t,r_t,a_s,r_s）の固定と整合チェック）
- `output/quantum/nuclear_effective_potential_square_well.png`（最小 ansatz（square-well）での `(B,a_t)` 拘束→`r_t` 予言と棄却条件）
- `output/quantum/nuclear_effective_potential_core_well.png`（core+well ansatz：triplet を `(B,a_t,r_t)` で拘束し、`v2t` と singlet `(r_s,v2s)` を予言して棄却条件を固定）
- `output/quantum/nuclear_effective_potential_finite_core_well.png`（finite core+well ansatz：追加自由度（Vc）を入れても解が `Rc→0,Vc→0` に退化し、singlet 予言が整合しないことを固定）
- `output/quantum/nuclear_effective_potential_two_range.png`（2-range ansatz：triplet を拘束し、singlet を同時拘束して棄却条件を固定）
- `output/quantum/nuclear_effective_potential_two_range_fit_as_rs.png`（2-range ansatz：singlet を `a_s,r_s` で拘束し、`v2s` の差分予測で棄却条件を固定）
- `output/quantum/nuclear_effective_potential_repulsive_core_two_range.png`（repulsive core + 2-range：符号構造を追加しても `v2s` の符号が救えないことを固定）
- `output/quantum/nuclear_effective_potential_pion_constrained_signed_v2.png`（λ_π 制約下：`v2s` が正に出る問題の可視化（Step 7.13.3））
- `output/quantum/nuclear_effective_potential_pion_constrained_three_range_tail.png`（λ_π 制約下：3-range tail 粗視化（shared tail）でも `v2s<0` が再現できず棄却（Step 7.13.4））
- `output/quantum/nuclear_effective_potential_pion_constrained_barrier_tail.png`（λ_π 制約下：tail を barrier+tail に分割（mean-preserving）しても `v2s<0` が再現できず棄却（Step 7.13.5））
- `output/quantum/nuclear_effective_potential_pion_constrained_barrier_tail_k_scan.png`（λ_π 制約下：barrier_height_factor k を走査しても `v2s<0` が救えず棄却（Step 7.13.6））
- `output/quantum/nuclear_effective_potential_pion_constrained_barrier_tail_q_scan.png`（λ_π 制約下：tail_depth_factor q を走査（free tail depth）しても包絡一致に至らず棄却（Step 7.13.7））
- `output/quantum/nuclear_effective_potential_pion_constrained_barrier_tail_kq_scan.png`（λ_π 制約下：global (k,q) の 2自由度走査で包絡一致（eq18–eq19）に到達（Step 7.13.8））
- `output/quantum/nuclear_effective_potential_pion_constrained_barrier_tail_kq_scan_v2t.png`（λ_π 制約下：拡張 (k,q) 探索で `v2t` を包絡へ入れることを試みるが、`r_s` が崩壊して no-go を固定（Step 7.13.8.1））
- `output/quantum/nuclear_effective_potential_pion_constrained_barrier_tail_channel_split_kq_scan.png`（λ_π 制約下：チャネル依存 (k,q) 分離でも `r_s` 崩壊が解消せず no-go を固定（Step 7.13.8.2））
- `output/quantum/nuclear_effective_potential_pion_constrained_barrier_tail_channel_split_kq_scan_singlet_r1_scan.png`（λ_π 制約下：7.13.8.2 の (k_t,q_t)/(k_s,q_s) を凍結したまま singlet `R1_s/λπ` を走査すると、`r_s` を合わせれば `v2s` が外れ、`v2s` を合わせれば `r_s` が崩壊する（no-go 継続；Step 7.13.8.3））
- `output/quantum/nuclear_effective_potential_pion_constrained_barrier_tail_channel_split_kq_scan_singlet_r2_scan.png`（λ_π 制約下：`R1_s/λπ=0.45` を凍結し singlet `R2_s/λπ` を走査すると、`R2_s/λπ=1.5` で `r_s` と `v2s` の同時包絡整合に到達（Step 7.13.8.4））
- `output/quantum/nuclear_effective_potential_pion_constrained_barrier_tail_channel_split_kq_scan_triplet_barrier_fraction_scan.png`（λ_π 制約下：7.13.8.4 の singlet split を凍結し、triplet の barrier/tail 分割比（barrier_len_fraction_t）と k_t を走査して、`(a_t,r_t)` を許容幅内へ回復（Step 7.13.8.5））

---

## LLR（月レーザー測距）

- `output/llr/batch/llr_batch_summary.json`（主要数値）
- `output/llr/batch/llr_time_tag_selection_by_station.png`（time-tag 最適化：局ごとの tx/rx/mid 選択）
- `output/llr/llr_primary_tof_timeseries.png`（CRD Normal Point：TOF 時系列）
- `output/llr/llr_primary_range_timeseries.png`（CRD Normal Point：片道距離 時系列）
- `output/llr/out_llr/llr_primary_overlay_tof.png`（観測 vs モデル：TOF重ね合わせ）
- `output/llr/out_llr/llr_primary_residual.png`（残差時系列：観測 - モデル）
- `output/llr/out_llr/llr_primary_residual_compare.png`（モデル改善の効果：地球中心→観測局→反射器）
- `output/llr/batch/llr_outliers_overview.png`（外れ値の俯瞰）
- `output/llr/batch/llr_outliers_time_tag_sensitivity.png`（外れ値の time-tag 感度：tx/rx/mid の影響）
- `output/llr/batch/llr_outliers_target_mixing_sensitivity.png`（外れ値の ターゲット混入感度：現ターゲット vs 推定ターゲット）
- `output/llr/batch/llr_residual_distribution.png`（残差分布：観測 - P-model（inlierのみ））
- `output/llr/batch/llr_rms_improvement_overall.png`（モデル要素の寄与：RMSの改善量（要約））
- `output/llr/batch/llr_rms_ablations_overall.png`（切り分け（Shapiro / 対流圏 / 月回転モデル））
- `output/llr/batch/llr_rms_by_station_target.png`（残差RMS：観測局→反射器 station×reflector）
- `output/llr/batch/llr_rms_by_station_month.png`（残差RMS：局別・月別）
- `output/llr/batch/llr_rms_by_station_month_models.png`（局別×月別の残差RMS：モデル別）
- `output/llr/batch/llr_grsm_rms_by_target_month.png`（GRSM 残差RMS：反射器別の期間依存）
- `output/llr/batch/llr_grsm_rms_by_month_models.png`（GRSM 残差RMS：モデル別）
- `output/llr/batch/llr_apol_residual_vs_elevation.png`（APOL 残差 vs 仰角（SR+Tropo+Tide））
- `output/llr/batch/llr_grsm_residual_vs_elevation.png`（GRSM 残差 vs 仰角（SR+Tropo+Tide））
- `output/llr/batch/llr_matm_residual_vs_elevation.png`（MATM 残差 vs 仰角（SR+Tropo+Tide））
- `output/llr/batch/llr_wetl_residual_vs_elevation.png`（WETL 残差 vs 仰角（SR+Tropo+Tide））
- `output/llr/batch/llr_station_coord_delta_pos_eop.png`（局座標差分（pos+eop vs site log））
- `output/llr/coord_compare/llr_station_coords_rms_compare.png`（局座標ソース比較（RMS））
- `output/llr/coord_compare/llr_grsm_monthly_rms_pos_eop_vs_slrlog.png`（GRSM 月別RMS（局座標ソース比較））
- `output/llr/batch/llr_shapiro_ablations_overall.png`（重力遅延の寄与（Shapiroの拡張））
- `output/llr/batch/llr_tide_ablations_overall.png`（潮汐の寄与（固体潮汐/海洋荷重/月体潮汐））

---

## Cassini（太陽会合）

- `output/cassini/cassini_fig2_overlay_full.png`（Cassini SCE1: ドップラー y(t) 観測 vs P-model（β=1））
- `output/cassini/cassini_fig2_overlay_zoom10d.png`（Cassini SCE1: ドップラー y(t)（±10日, 観測 vs P-model））
- `output/cassini/cassini_pds_vs_digitized.png`（Cassini SCE1: PDS一次データ（処理後） vs 論文図デジタイズ）
- `output/cassini/cassini_fig2_residuals.png`（Cassini SCE1: 残差（観測 - P-model））
- `output/cassini/cassini_beta_sweep_rmse.png`（Cassini: βスイープ（RMSE感度））
- `output/cassini/cassini_fig2_metrics.csv`（RMSE/相関）

---

## Viking（Shapiro遅延）

- `output/viking/viking_p_model_vs_measured_no_arrow.png`（Viking 1976 太陽会合: 往復Shapiro遅延（P-model vs 文献代表値））
- `output/viking/viking_shapiro_result.csv`（数値）

---

## Mercury（近日点移動）

- `output/mercury/mercury_precession_metrics.json`（数値）
- `output/mercury/mercury_orbit.png`（水星: 軌道と近日点移動（P-modelシミュレーション））

---

## GPS（時刻残差）

- `output/gps/gps_compare_metrics.json`（数値）
- `output/gps/gps_rms_compare.png`（GPS: 残差RMS（放送暦BRDC vs P-model, 観測IGS基準））
- `output/gps/gps_residual_compare_G01.png`（GPS 時計残差：G01（観測IGSに対する比較））
- `output/gps/gps_clock_residuals_all_31.png`（GPS 放送暦 時計残差（BRDC - IGS, 全衛星））
- `output/gps/gps_relativistic_correction_G02.png`（GPS: 相対補正（周期成分）の比較例（標準式 vs P-model））

---

## EHT（ブラックホール影）

- `output/eht/eht_shadow_compare.png`（EHT: リング直径（公表値） vs P-model影直径（κ=1仮定））
- `output/eht/eht_m87_persistent_shadow_ring_diameter.png`（EHT: M87* の multi-epoch（2017/2018）リング直径の整合性）
- `output/eht/eht_shadow_zscores.png`（EHT: ずれのzスコア（κ=1の仮定））
- `output/eht/eht_kappa_fit.png`（EHT: κ_fit（観測リング直径と理論シャドウ直径から逆算した κ））
- `output/eht/eht_kappa_precision_required.png`（EHT: 3σ判別に必要な κ 精度（相対不確かさの目安））
- `output/eht/eht_kappa_tradeoff.png`（EHT: 3σ判別の誤差予算（ring σ と κσ のトレードオフ））
- `output/eht/eht_delta_precision_required.png`（EHT: δ（Schwarzschild shadow deviation）の必要精度（参考））
- `output/eht/eht_shadow_differential.png`（EHT: 差分予測（影直径係数: P-model vs GR））
- `output/eht/eht_shadow_systematics.png`（EHT: 系統不確かさ（κ・Kerrスピン依存など））
- `output/eht/eht_kappa_error_budget.png`（EHT: κ誤差予算（scale indicators；較正/散乱/画像化/モデル依存プロキシの俯瞰））
- `output/eht/eht_kerr_shadow_coeff_grid.png`（EHT: Kerr shadow 係数の (a*,inc) 依存（reference systematic））
- `output/eht/eht_kerr_shadow_coeff_definition_sensitivity.png`（EHT: Kerr shadow 係数の定義依存（effective diameter）感度）
- `output/eht/eht_ring_morphology.png`（EHT: リング形状指標（幅 W/d と非対称 A））
- `output/eht/eht_sgra_paper5_pass_fail_tables_fail_fractions.png`（EHT: Paper V（放射モデル）Pass/Fail の失敗率（支配制約））
- `output/eht/eht_sgra_paper5_pass_fail_global_constraint_fail_fractions.png`（EHT: Paper V（放射モデル）Pass/Fail の全表集計（支配制約のglobal fail fraction））
- `output/eht/eht_sgra_paper5_near_passing_failed_constraints.png`（EHT: Paper V（放射モデル）near-passing（fail_one/fail_none）の「最後に残る制約」頻度）
- `output/eht/eht_sgra_paper5_m3_nir_reconnection_conditions.png`（EHT: Paper V（放射モデル）M3 / 2.2 μm の前提分解と reconnection 条件）
- `output/eht/eht_sgra_paper5_m3_historical_distribution_values_ecdf.png`（EHT: Paper V（放射モデル）historical distribution（mi3値; n=42/42）ECDF）
- `output/eht/eht_scattering_budget.png`（EHT: 散乱blurのスケール（Sgr A*）とリング直径/リング幅の比較）
- `output/eht/eht_refractive_scattering_limits.png`（EHT: 屈折散乱による像ゆらぎ（Zhu 2018）と必要精度の比較）

---

## XRISM（X線高分解能スペクトル：Resolve）

- `output/xrism/xrism_resolve_summary.png`（XRISM Resolve: BH/AGN（吸収線→速度）と銀河団（輝線→Δv/σ_v）のサマリ（一次データ再解析の入口））
- `output/xrism/fek_relativistic_broadening_isco_constraints.png`（Fe-Kα 相対論的広がり（GX 339-4; XMM+NuSTAR）: 内縁半径 r_in（ISCO proxy）の横断比較。統計誤差（細線）と系統＋統計（太線）を同時に示す）
- `output/xrism/000125000__spectrum_qc.png`（XRISM Resolve: スペクトル抽出QC（PHA/RMF/ARF、gain/ビニング/帯域の基本診断；obsidごとに固定出力））

---

## Pulsar（二重パルサー）

- `output/pulsar/binary_pulsar_orbital_decay.png`（二重パルサー: 軌道減衰（観測/GRの一致度））
- `output/pulsar/binary_pulsar_orbital_decay_metrics.json`（数値）

---

## GW（重力波）

- `output/gw/gw150914_chirp_phase.png`（GW150914: chirp位相（周波数上昇）の整合性チェック）
- `output/gw/gw150914_waveform_compare.png`（GW150914: 波形（bandpass後） vs 単純モデル（四重極）テンプレート）
- `output/gw/gw150914_chirp_phase_metrics.json`（数値）
- `output/gw/gw151226_chirp_phase.png`（GW151226: chirp位相（周波数上昇）の整合性チェック）
- `output/gw/gw151226_waveform_compare.png`（GW151226: 波形（bandpass後） vs 単純モデル（四重極）テンプレート）
- `output/gw/gw151226_chirp_phase_metrics.json`（数値）
- `output/gw/gw170104_chirp_phase.png`（GW170104: chirp位相（周波数上昇）の整合性チェック）
- `output/gw/gw170104_waveform_compare.png`（GW170104: 波形（bandpass後） vs 単純モデル（四重極）テンプレート）
- `output/gw/gw170104_chirp_phase_metrics.json`（数値）
- `output/gw/gw170817_chirp_phase.png`（GW170817: chirp位相（周波数上昇）の整合性チェック）
- `output/gw/gw170817_waveform_compare.png`（GW170817: 波形（bandpass後） vs 単純モデル（四重極）テンプレート）
- `output/gw/gw170817_chirp_phase_metrics.json`（数値）
- `output/gw/gw190425_chirp_phase.png`（GW190425: chirp位相（周波数上昇）の整合性チェック）
- `output/gw/gw190425_waveform_compare.png`（GW190425: 波形（bandpass後） vs 単純モデル（四重極）テンプレート）
- `output/gw/gw190425_chirp_phase_metrics.json`（数値）
- `output/gw/gw200115_042309_chirp_phase.png`（GW200115_042309: chirp位相（周波数上昇）の整合性チェック）
- `output/gw/gw200115_042309_waveform_compare.png`（GW200115_042309: 波形（bandpass/whiten後） vs 単純モデル（四重極）テンプレート）
- `output/gw/gw200115_042309_chirp_phase_metrics.json`（数値）
- `output/gw/gw200129_065458_chirp_phase.png`（GW200129_065458: chirp位相（周波数上昇）の整合性チェック）
- `output/gw/gw200129_065458_waveform_compare.png`（GW200129_065458: 波形（bandpass後） vs 単純モデル（四重極）テンプレート）
- `output/gw/gw200129_065458_chirp_phase_metrics.json`（数値）
- `output/gw/gw200224_222234_chirp_phase.png`（GW200224_222234: chirp位相（周波数上昇）の整合性チェック）
- `output/gw/gw200224_222234_waveform_compare.png`（GW200224_222234: 波形（bandpass後） vs 単純モデル（四重極）テンプレート）
- `output/gw/gw200224_222234_chirp_phase_metrics.json`（数値）
- `output/gw/gw191216_213338_chirp_phase.png`（GW191216_213338: chirp位相（周波数上昇）の整合性チェック）
- `output/gw/gw191216_213338_waveform_compare.png`（GW191216_213338: 波形（bandpass後） vs 単純モデル（四重極）テンプレート）
- `output/gw/gw191216_213338_chirp_phase_metrics.json`（数値）
- `output/gw/gw200311_115853_chirp_phase.png`（GW200311_115853: chirp位相（周波数上昇）の整合性チェック）
- `output/gw/gw200311_115853_waveform_compare.png`（GW200311_115853: 波形（bandpass後） vs 単純モデル（四重極）テンプレート）
- `output/gw/gw200311_115853_chirp_phase_metrics.json`（数値）
- `output/gw/gw_multi_event_summary.png`（重力波: 複数イベントの要約（R^2 と match））
- `output/gw/gw_multi_event_summary_metrics.json`（数値）
- `output/gw/gw250114_ringdown_qnm_fit.png`（GW250114: ringdown の QNM（周波数/減衰時間）fit と残差の要約）
- `output/gw/gw250114_area_theorem_test.png`（GW250114: Area theorem 検証（前後地平面積の差の z-score；4.8σ））
- `output/gw/gw250114_imr_consistency.png`（GW250114: IMR consistency test（inspiral vs post-inspiral の整合性））

---

## Theory（差分予測）

- `output/theory/solar_light_deflection.png`（太陽重力による光の偏向：α(b) と 観測γ（一次ソース） vs P-model）  
- `output/theory/gravitational_redshift_experiments.png`（重力赤方偏移：観測 vs P-model（一次ソース））
- `output/theory/frame_dragging_experiments.png`（回転（フレームドラッグ）：観測 vs P-model（一次ソース））
- `output/theory/dynamic_p_quadrupole_scalings.png`（動的P（放射・波動）：四重極則のスケーリング（概念スケール））
- `output/theory/dynamic_p_quadrupole_scalings_metrics.json`（数値）
- `output/theory/delta_saturation_constraints.png`（速度飽和δ: 既知の高γ観測と矛盾しない条件）
- `output/theory/delta_saturation_constraints.json`（数値）

---

## Cosmology（宇宙論）

（本文 2.5 の構成）
- 2.5.1：BAO一次統計の詳細（一次統計 ξℓ / ξ(s,μ) から再導出）
- 2.5.2：宇宙論 I（距離指標と独立な検証：SN time dilation / CMB T(z) / AP）
- 2.5.3：宇宙論 II（距離指標依存と BAO の前提：DDR/Tolman/誤差予算/再接続）

- `output/cosmology/cosmology_redshift_pbg.png`（宇宙膨張なしで赤方偏移をPで説明：背景Pの時間変化）
- `output/cosmology/cosmology_redshift_pbg_metrics.json`（数値）
- `output/cosmology/cosmology_observable_scalings.png`（宇宙論：観測量スケーリング（距離二重性η / Tolman表面輝度）の差分予測）
- `output/cosmology/cosmology_observable_scalings_metrics.json`（数値）
- `output/cosmology/cosmology_sn_time_dilation_constraints.png`（宇宙論：SN time dilation の一次ソース制約（p_t））
- `output/cosmology/cosmology_sn_time_dilation_constraints_metrics.json`（数値）
- `output/cosmology/cosmology_cmb_temperature_scaling_constraints.png`（宇宙論：CMB温度スケーリング T(z) の一次ソース制約）
- `output/cosmology/cosmology_cmb_temperature_scaling_constraints_metrics.json`（数値）
- `output/cosmology/cosmology_alcock_paczynski_constraints.png`（宇宙論：Alcock-Paczynski（F_AP）制約と静的背景Pモデル比較）
- `output/cosmology/cosmology_alcock_paczynski_constraints_metrics.json`（数値）
- `output/cosmology/cosmology_ddr_pmodel_eta_constraints.png`（宇宙論：距離二重性η^(P)指標の観測制約（前提監査））
- `output/cosmology/cosmology_distance_duality_constraints.png`（宇宙論：距離二重性ηの観測制約（棄却条件））
- `output/cosmology/cosmology_distance_duality_constraints_metrics.json`（数値）
- `output/cosmology/cosmology_distance_duality_source_sensitivity.png`（宇宙論：DDR一次ソース依存（距離指標で結論がどれだけ変わるか））
- `output/cosmology/cosmology_distance_duality_source_sensitivity_metrics.json`（数値）
- `output/cosmology/cosmology_distance_duality_internal_consistency.png`（宇宙論：DDR制約の内部整合性（ε0の相互矛盾をペア差の|z|で可視化））
- `output/cosmology/cosmology_distance_duality_anchor_sensitivity.png`（宇宙論：DDRのη0パラメータ化→ε0換算のz_ref感度（定義差の寄与））
- `output/cosmology/cosmology_distance_duality_systematics_envelope.png`（宇宙論：DDRのモデル差を系統幅として取り込み、棄却度の頑健性を再評価）
- `output/cosmology/cosmology_distance_duality_internal_consistency_metrics.json`（数値）
- `output/cosmology/cosmology_distance_duality_systematics_envelope_metrics.json`（数値）
- `output/cosmology/cosmology_distance_indicator_reach_limit.png`（宇宙論：距離指標の“到達限界”の整理（必要な |Δμ|/|τ| の z 依存））
- `output/cosmology/cosmology_distance_indicator_reach_limit_metrics.json`（数値）
- `output/cosmology/cosmology_distance_indicator_error_budget.png`（宇宙論：距離指標（SNe Ia/BAO）の誤差予算（z依存の目安）と“到達限界”の接続）
- `output/cosmology/cosmology_distance_indicator_error_budget_metrics.json`（数値）
- `output/cosmology/cosmology_distance_indicator_error_budget_sensitivity.png`（宇宙論：距離指標の誤差予算の感度（SNeサンプル/共分散の扱いで z_limit がどれだけ動くか））
- `output/cosmology/cosmology_distance_indicator_error_budget_sensitivity_metrics.json`（数値）
- `output/cosmology/cosmology_distance_indicator_rederivation_requirements.png`（宇宙論：距離指標の再導出要件（静的背景Pで DDR 制約を回避するために何を捨てる/置き換える必要があるか））
- `output/cosmology/cosmology_distance_indicator_rederivation_requirements_metrics.json`（数値）
- `output/cosmology/cosmology_distance_indicator_rederivation_candidate_search.png`（宇宙論：距離指標の再導出候補探索（DDR/BAO/独立プローブを同時に満たす組合せの有無））
- `output/cosmology/cosmology_distance_indicator_rederivation_candidate_search_metrics.json`（数値）
- `output/cosmology/cosmology_distance_indicator_rederivation_limiting_summary.png`（宇宙論：再導出候補探索の支配拘束（limiting）の一覧（DDR一次ソースごと））
- `output/cosmology/cosmology_distance_indicator_rederivation_limiting_summary_metrics.json`（数値）
- `output/cosmology/cosmology_distance_indicator_rederivation_selected_sources.png`（宇宙論：再導出候補探索で選択されやすい一次ソース拘束（α / s_L）の頻度）
- `output/cosmology/cosmology_distance_indicator_rederivation_selected_sources_metrics.json`（数値）
- `output/cosmology/cosmology_distance_indicator_rederivation_policy_sensitivity.png`（宇宙論：一次ソース選択ルールの感度（scan vs tightest_fixed））
- `output/cosmology/cosmology_distance_indicator_rederivation_policy_sensitivity_metrics.json`（数値）
- `output/cosmology/cosmology_distance_indicator_rederivation_global_prior_search.png`（宇宙論：再導出候補探索の頑健性（α と s_L を単一priorに固定した場合の全DDR整合; worst max|z|））
- `output/cosmology/cosmology_distance_indicator_rederivation_global_prior_search_metrics.json`（数値）
- `output/cosmology/cosmology_distance_indicator_rederivation_global_prior_search_bao_sigma_scan.png`（宇宙論：単一prior（α,s_L）を固定した場合の整合度と BAO(s_R) 緩和（σ→fσ）の関係）
- `output/cosmology/cosmology_distance_indicator_rederivation_global_prior_search_bao_sigma_scan_metrics.json`（数値）
- `output/cosmology/cosmology_distance_indicator_rederivation_candidate_search_bao_sigma_scan.png`（宇宙論：再導出候補探索の感度（BAO(s_R) を soft constraint とした σスケールの影響））
- `output/cosmology/cosmology_distance_indicator_rederivation_candidate_search_bao_sigma_scan_metrics.json`（数値）
- `output/cosmology/cosmology_distance_indicator_rederivation_bao_relax_thresholds.png`（宇宙論：再導出候補探索の全DDR行まとめ（BAO(s_R) を soft constraint としたときの max|z| 行列））
- `output/cosmology/cosmology_distance_indicator_rederivation_bao_relax_thresholds_metrics.json`（数値）
- `output/cosmology/cosmology_distance_indicator_rederivation_bao_mode_sensitivity.png`（宇宙論：再導出候補探索の感度（BAO(s_R) fit の共分散モードによる max|z| 行列））
- `output/cosmology/cosmology_distance_indicator_rederivation_bao_mode_sensitivity_metrics.json`（数値）
- `output/cosmology/cosmology_distance_indicator_rederivation_bao_mode_global_prior_sigma_scan.png`（宇宙論：BAO fit 前提（共分散モード）ごとの「単一priorで1σ同時整合に必要な緩和量 f」）
- `output/cosmology/cosmology_distance_indicator_rederivation_bao_mode_global_prior_sigma_scan_metrics.json`（数値）
- `output/cosmology/cosmology_distance_indicator_rederivation_bao_survey_global_prior_sigma_scan.png`（宇宙論：BAO一次ソース（BOSS/eBOSS/併合）ごとの「単一priorで1σ同時整合に必要な緩和量 f」）
- `output/cosmology/cosmology_distance_indicator_rederivation_bao_survey_global_prior_sigma_scan_metrics.json`（数値）
- `output/cosmology/cosmology_distance_indicator_rederivation_bao_survey_leave_one_out_global_prior_sigma_scan.png`（宇宙論：BAO併合fit の leave-one-out（点を1つ除外）ごとの「単一priorで1σ同時整合に必要な緩和量 f」）
- `output/cosmology/cosmology_distance_indicator_rederivation_bao_survey_leave_one_out_global_prior_sigma_scan_metrics.json`（数値）
- `output/cosmology/cosmology_distance_indicator_rederivation_candidate_matrix.png`（宇宙論：再導出候補マトリクス（α×s_L の全組合せで max|z| を可視化））
- `output/cosmology/cosmology_distance_indicator_rederivation_candidate_matrix_metrics.json`（数値）
- `output/cosmology/cosmology_ddr_reconnection_conditions.png`（宇宙論：DDR再接続に必要な条件（α/s_L/s_R）の要約と一次ソース拘束の比較）
- `output/cosmology/cosmology_ddr_reconnection_conditions_metrics.json`（数値）
- `output/cosmology/cosmology_reconnection_parameter_space.png`（宇宙論：静的背景PでDDR（ε0）を回復する“必要補正”のパラメータ空間（不透明度/進化の組合せ））
- `output/cosmology/cosmology_reconnection_parameter_space_metrics.json`（数値）
- `output/cosmology/cosmology_reconnection_plausibility.png`（宇宙論：DDR再接続に必要な補正（α/s_L/r_drag）と一次ソース拘束の比較）
- `output/cosmology/cosmology_reconnection_plausibility_metrics.json`（数値）
- `output/cosmology/cosmology_reconnection_required_ruler_evolution.png`（宇宙論：一次ソース拘束（不透明度α・標準光源進化s_L）込みでDDRを満たすために必要な標準定規進化s_R）
- `output/cosmology/cosmology_reconnection_required_ruler_evolution_metrics.json`（数値）
- `output/cosmology/cosmology_reconnection_required_ruler_evolution_independent.png`（宇宙論：独立一次ソース版（BAO/CMBを使わない α と s_L の拘束）でDDRを満たすために必要な標準定規進化s_R）
- `output/cosmology/cosmology_reconnection_required_ruler_evolution_independent_metrics.json`（数値）
- `output/cosmology/cosmology_bao_scaled_distance_fit.png`（宇宙論：BAO（BOSS DR12）の (D_M,H) 出力から s_R をfitし、DDR必要値との張力を可視化）
- `output/cosmology/cosmology_bao_scaled_distance_fit_metrics.json`（数値）
- `output/cosmology/cosmology_bao_scaled_distance_fit_independent.png`（宇宙論：BAO（BOSS DR12）の (D_M,H) 出力から s_R をfitし、独立一次ソース版のDDR必要値との張力を可視化）
- `output/cosmology/cosmology_bao_scaled_distance_fit_independent_metrics.json`（数値）
- `output/cosmology/cosmology_bao_scaled_distance_fit_sensitivity.png`（宇宙論：BAO(s_R) 推定の共分散感度（full/diag/ブロック）と、再導出候補探索（best_independent）への影響）
- `output/cosmology/cosmology_bao_scaled_distance_fit_sensitivity_metrics.json`（数値）
- `output/cosmology/cosmology_bao_distance_ratio_fit.png`（宇宙論：BAOの多系統比較（BOSS/eBOSS）の D_M/r_d と D_H/r_d を fit して s_R を推定）
- `output/cosmology/cosmology_bao_distance_ratio_fit_residuals.png`（宇宙論：BAO多系統の併合fitで、点ごとの残差（√χ²寄与）を可視化）
- `output/cosmology/cosmology_bao_distance_ratio_fit_leave_one_out.png`（宇宙論：BAO多系統の併合fitで、leave-one-out（点を1つ落とす）感度を可視化）
- `output/cosmology/cosmology_bao_distance_ratio_fit_metrics.json`（数値）
- `output/cosmology/cosmology_bao_xi_multipole_peakfit.png`（宇宙論：BAO一次統計（BOSS DR12 post-recon ξℓ）からのピークfit（smooth+peak）で、AP異方（ε）を評価）
- `output/cosmology/cosmology_bao_xi_multipole_peakfit_metrics.json`（数値）
- `output/cosmology/cosmology_bao_xi_multipole_peakfit_pre_recon.png`（宇宙論：BAO一次統計（BOSS DR12 pre-recon ξℓ）でのクロスチェック（smooth+peak））
- `output/cosmology/cosmology_bao_xi_multipole_peakfit_pre_recon_metrics.json`（数値）
- `output/cosmology/cosmology_bao_catalog_vs_published_crosscheck.png`（宇宙論：BAO一次情報（galaxy+random）の catalog-based ξℓ と、公開 multipoles（post/pre-recon）の ε（AP warping）を突き合わせるクロスチェック）
- `output/cosmology/cosmology_bao_catalog_vs_published_crosscheck_metrics.json`（数値）
- `output/cosmology/cosmology_bao_catalog_vs_published_multipoles_overlay.png`（宇宙論：公開 multipoles（Ross post-recon / Satpathy pre-recon）と catalog-based ξℓ（pre-recon + recon(iso); Corrfunc）を曲線（s²ξ0/ξ2）で重ねて比較）
- `output/cosmology/cosmology_bao_catalog_vs_published_multipoles_overlay_metrics.json`（数値）
- `output/cosmology/cosmology_bao_catalog_peakfit_cmass_combined.png`（宇宙論：BAO一次情報（galaxy+random, CMASS合算）の peakfit（smooth+peak）で ε（AP warping）を推定）
- `output/cosmology/cosmology_bao_catalog_peakfit_cmass_combined_metrics.json`（数値）
- `output/cosmology/cosmology_bao_catalog_peakfit_cmass_combined__recon_grid_iso.png`（宇宙論：BAO一次情報（galaxy+random, CMASS合算）に簡易reconstruction（grid; iso）を適用した peakfit（smooth+peak））
- `output/cosmology/cosmology_bao_catalog_peakfit_cmass_combined__recon_grid_iso_metrics.json`（数値）
- `output/cosmology/cosmology_bao_catalog_peakfit_lowz_combined.png`（宇宙論：BAO一次情報（galaxy+random, LOWZ合算）の peakfit（smooth+peak）で ε（AP warping）を推定）
- `output/cosmology/cosmology_bao_catalog_peakfit_lowz_combined_metrics.json`（数値）
- `output/cosmology/cosmology_bao_catalog_peakfit_lrgpcmass_rec_combined.png`（宇宙論：BAO一次情報（galaxy+random, eBOSS DR16 LRGpCMASS recon）の peakfit（smooth+peak）で ε（AP warping）を推定）
- `output/cosmology/cosmology_bao_catalog_peakfit_lrgpcmass_rec_combined_metrics.json`（数値）
- `output/cosmology/cosmology_bao_catalog_peakfit_lrgpcmass_rec_north.png`（宇宙論：eBOSS DR16 LRGpCMASS recon の north（NGC）分割；系統切り分け）
- `output/cosmology/cosmology_bao_catalog_peakfit_lrgpcmass_rec_north_metrics.json`（数値）
- `output/cosmology/cosmology_bao_catalog_peakfit_lrgpcmass_rec_south.png`（宇宙論：eBOSS DR16 LRGpCMASS recon の south（SGC）分割；系統切り分け）
- `output/cosmology/cosmology_bao_catalog_peakfit_lrgpcmass_rec_south_metrics.json`（数値）
- `output/cosmology/cosmology_bao_catalog_peakfit_qso_combined.png`（宇宙論：BAO一次情報（galaxy+random, eBOSS DR16 QSO）の peakfit（smooth+peak）で ε（AP warping）を推定）
- `output/cosmology/cosmology_bao_catalog_peakfit_qso_combined_metrics.json`（数値）
- `output/cosmology/cosmology_bao_catalog_peakfit_qso_north.png`（宇宙論：eBOSS DR16 QSO の north（NGC）分割；系統切り分け）
- `output/cosmology/cosmology_bao_catalog_peakfit_qso_north_metrics.json`（数値）
- `output/cosmology/cosmology_bao_catalog_peakfit_qso_south.png`（宇宙論：eBOSS DR16 QSO の south（SGC）分割；系統切り分け）
- `output/cosmology/cosmology_bao_catalog_peakfit_qso_south_metrics.json`（数値）
- `output/cosmology/cosmology_bao_catalog_peakfit_lrg_combined__w_desi_default_ms_off.png`（宇宙論：BAO一次情報（galaxy+random, DESI DR1 LRG）の peakfit（smooth+peak）で ε（AP warping）を推定）
- `output/cosmology/cosmology_bao_catalog_peakfit_lrg_combined__w_desi_default_ms_off_metrics.json`（数値）
- `output/cosmology/cosmology_bao_catalog_peakfit_lrg_north__w_desi_default_ms_off.png`（宇宙論：DESI DR1 LRG の north（NGC）分割；系統切り分け）
- `output/cosmology/cosmology_bao_catalog_peakfit_lrg_north__w_desi_default_ms_off_metrics.json`（数値）
- `output/cosmology/cosmology_bao_catalog_peakfit_lrg_south__w_desi_default_ms_off.png`（宇宙論：DESI DR1 LRG の south（SGC）分割；系統切り分け）
- `output/cosmology/cosmology_bao_catalog_peakfit_lrg_south__w_desi_default_ms_off_metrics.json`（数値）
- `output/cosmology/cosmology_bao_catalog_peakfit_lrg_combined__w_desi_default_ms_off_y1bins.png`（宇宙論：DESI DR1 BAO 公式 bin（LRG1/LRG2: 0.4–0.6/0.6–0.8）に揃えた ξℓ→peakfit（diag））
- `output/cosmology/cosmology_bao_catalog_peakfit_lrg_combined__w_desi_default_ms_off_y1bins_metrics.json`（数値）
- `output/cosmology/cosmology_bao_catalog_peakfit_lrg_combined__w_desi_default_ms_off_y1bins_full_r0__jk_cov_both_full_r0.png`（宇宙論：DESI DR1 BAO 公式 bin（LRG1/LRG2）に揃えた ξℓ→peakfit（sky jackknife cov; dv+cov; full random））
- `output/cosmology/cosmology_bao_catalog_peakfit_lrg_combined__w_desi_default_ms_off_y1bins_full_r0__jk_cov_both_full_r0_metrics.json`（数値）
- `output/cosmology/cosmology_desi_dr1_bao_y1data_eps_crosscheck__w_desi_default_ms_off_y1bins.png`（宇宙論：DESI DR1 BAO（Y1data: D_M/r_d, D_H/r_d, corr）から ε_expected を算出し、catalog peakfit ε（jackknife）と比較）
- `output/cosmology/cosmology_desi_dr1_bao_y1data_eps_crosscheck__w_desi_default_ms_off_y1bins_metrics.json`（数値）
- `output/cosmology/cosmology_desi_dr1_bao_cov_shrinkage_sweep__w_desi_default_ms_off_y1bins_lrg3elg1_reservoir_r0to17_mix_rescale__jk_cov_ra_dec_per_cap_unwrapped_n48_dec4_ra6_band0.png`（宇宙論：DESI DR1 LRG3+ELG1 の dv+cov（自前生成）に対して、covariance shrinkage λ のスイープで ε_expected cross-check（z_score_combined）の安定性を評価）
- `output/cosmology/cosmology_desi_dr1_bao_cov_shrinkage_sweep__w_desi_default_ms_off_y1bins_lrg3elg1_reservoir_r0to17_mix_rescale__jk_cov_ra_dec_per_cap_unwrapped_n48_dec4_ra6_band0.json`（数値）
- `output/cosmology/cosmology_desi_dr1_bao_cov_shrinkage_sweep__w_desi_default_ms_off_y1bins_lrg3elg1_reservoir_r0to17_mix_rescale__jk_cov_ra_dec_per_cap_unwrapped_n48_dec4_ra6_band0.csv`（数値）
- `output/cosmology/cosmology_desi_dr1_bao_cov_shrinkage_sweep__w_desi_vac_lya_corr_v1p0_mubin_comb__vac_cov_smooth3.png`（宇宙論：DESI DR1 VAC（Lyα correlations; Lyα QSO）に対して、covariance shrinkage λ のスイープで ε_expected cross-check（z_score_combined）の安定性を評価）
- `output/cosmology/cosmology_desi_dr1_bao_cov_shrinkage_sweep__w_desi_vac_lya_corr_v1p0_mubin_comb__vac_cov_smooth3.json`（数値）
- `output/cosmology/cosmology_desi_dr1_bao_cov_shrinkage_sweep__w_desi_vac_lya_corr_v1p0_mubin_comb__vac_cov_smooth3.csv`（数値）
- `output/cosmology/cosmology_desi_dr1_bao_promotion_check.png`（宇宙論：DESI DR1 multi-tracer の “screening→確証” 昇格判定の要約図（passing tracers の z_score_combined を λ スイープで可視化））
- `output/cosmology/cosmology_desi_dr1_bao_promotion_check_public.png`（宇宙論：DESI DR1 multi-tracer の “screening→確証” 昇格判定の要約図（paper用の簡略版））
- `output/cosmology/cosmology_desi_dr1_bao_promotion_check.json`（宇宙論：DESI DR1 multi-tracer の “screening→確証” 昇格判定（stable |z|≥3; min_tracers=2）の固定出力）
- `output/cosmology/cosmology_bao_catalog_peakfit_settings_sensitivity__lrg_combined__w_desi_default_ms_off_y1bins.png`（宇宙論：DESI LRG1/LRG2 の ε が peakfit 設定（r_range/テンプレート/quad_weight 等）でどれだけ動くか（jackknife cov 固定））
- `output/cosmology/cosmology_bao_catalog_peakfit_settings_sensitivity__lrg_combined__w_desi_default_ms_off_y1bins_metrics.json`（数値）
- `output/cosmology/cosmology_bao_catalog_coordinate_spec_sensitivity__lrg_combined__w_desi_default_ms_off_y1bins.png`（宇宙論：DESI LRG1/LRG2 の ε が座標化仕様（z_source / lcdm距離積分設定）でどれだけ動くか（diag））
- `output/cosmology/cosmology_bao_catalog_coordinate_spec_sensitivity__lrg_combined__w_desi_default_ms_off_y1bins_metrics.json`（数値）
- `output/cosmology/cosmology_bao_catalog_wedge_anisotropy__w_desi_default_ms_off_y1bins.png`（宇宙論：DESI LRG1/LRG2（y1bins）の μ-wedge（横/縦）ピーク差（s⊥, s∥）による異方サマリ（ε_proxy））
- `output/cosmology/cosmology_bao_catalog_wedge_anisotropy__w_desi_default_ms_off_y1bins_metrics.json`（数値）
- `output/cosmology/cosmology_bao_catalog_peak_summary__w_desi_default_ms_off_y1bins.png`（宇宙論：DESI LRG1/LRG2（y1bins）のピーク指標（ξ0/ξ2 feature）と z依存サマリ）
- `output/cosmology/cosmology_bao_catalog_peak_summary__w_desi_default_ms_off_y1bins_metrics.json`（数値）
- `output/cosmology/cosmology_bao_catalog_peakfit_lowz_combined__recon_grid_iso.png`（宇宙論：BAO一次情報（galaxy+random, LOWZ合算）に簡易reconstruction（grid; iso）を適用した peakfit（smooth+peak））
- `output/cosmology/cosmology_bao_catalog_peakfit_lowz_combined__recon_grid_iso_metrics.json`（数値）
- `output/cosmology/cosmology_bao_catalog_peakfit_caps_summary__recon_grid_iso.png`（宇宙論：BAO一次情報（galaxy+random, recon(iso)）の peakfit ε を NGC/SGC 分割で比較（系統切り分け））
- `output/cosmology/cosmology_bao_catalog_peakfit_caps_summary__recon_grid_iso_metrics.json`（数値）
- `output/cosmology/cosmology_bao_catalog_peakfit_cmasslowztot_combined_zbinonly__prerecon.png`（宇宙論：BAO一次情報（galaxy+random, z-bin, pre-recon）の peakfit（smooth+peak；Satpathy cov）で ε(z) を比較）
- `output/cosmology/cosmology_bao_catalog_peakfit_cmasslowztot_combined_zbinonly__prerecon_metrics.json`（数値）
- `output/cosmology/cosmology_bao_catalog_peakfit_cmasslowztot_combined_zbinonly__recon_grid_iso.png`（宇宙論：BAO一次情報（galaxy+random, z-bin）に簡易reconstruction（grid; iso）を適用した peakfit（smooth+peak））
- `output/cosmology/cosmology_bao_catalog_peakfit_cmasslowztot_combined_zbinonly__recon_grid_iso_metrics.json`（数値）
- `output/cosmology/cosmology_bao_catalog_peakfit_cmasslowztot_combined_zbinonly__recon_mw_multigrid_ani_mw_rw_boss_recon_rz_zmin0p2_zmax0p75_g512_bias1p85_f0p757.png`（宇宙論：BAO一次情報（galaxy+random, z-bin）に MW multigrid recon（ani; boss_recon重み）を適用した peakfit（smooth+peak；Ross cov；lcdm/pbg比較））
- `output/cosmology/cosmology_bao_catalog_peakfit_cmasslowztot_combined_zbinonly__recon_mw_multigrid_ani_mw_rw_boss_recon_rz_zmin0p2_zmax0p75_g512_bias1p85_f0p757_metrics.json`（数値）
- `output/cosmology/cosmology_bao_catalog_wedge_anisotropy.png`（宇宙論：BAO一次情報（galaxy+random）の μ-wedge（横/縦）ピーク差（s⊥, s∥）による異方サマリ）
- `output/cosmology/cosmology_bao_catalog_wedge_anisotropy_metrics.json`（数値）
- `output/cosmology/cosmology_bao_catalog_wedge_anisotropy__recon_grid_iso.png`（宇宙論：BAO一次情報（galaxy+random）に簡易reconstruction（grid; iso）を適用した μ-wedge 異方サマリ）
- `output/cosmology/cosmology_bao_catalog_wedge_anisotropy__recon_grid_iso_metrics.json`（数値）
- `output/cosmology/cosmology_bao_catalog_peak_summary.png`（宇宙論：BAO一次情報（galaxy+random）のピーク指標（ξ0/ξ2）と z依存サマリ）
- `output/cosmology/cosmology_bao_catalog_peak_summary_metrics.json`（数値）
- `output/cosmology/cosmology_bao_catalog_peak_summary__recon_grid_iso.png`（宇宙論：BAO一次情報（galaxy+random）に簡易reconstruction（grid; iso）を適用したピーク指標（ξ0/ξ2）と z依存サマリ）
- `output/cosmology/cosmology_bao_catalog_peak_summary__recon_grid_iso_metrics.json`（数値）
- `output/cosmology/cosmology_bao_recon_param_scan.png`（宇宙論：catalog-based ξℓ に対する簡易reconstruction（grid; iso）のパラメータ（平滑化スケール等）を振り、Ross post-recon への一致度をスキャン（RMSEに加え、Ross 公開 covariance から χ2/dof（xi0+xi2）も併記））
- `output/cosmology/cosmology_bao_recon_param_scan_metrics.json`（数値）
- `output/cosmology/cosmology_bao_recon_param_scan_ani.png`（宇宙論：catalog-based ξℓ に対する簡易reconstruction（grid; ani）のパラメータを振り、Ross post-recon への一致度をスキャン（RMSEに加え、Ross 公開 covariance から χ2/dof（xi0+xi2）も併記））
- `output/cosmology/cosmology_bao_recon_param_scan_ani_metrics.json`（数値）
- `output/cosmology/cosmology_bao_recon_gap_summary.png`（宇宙論：Ross post-recon の ξ2 ギャップについて、これまで試した recon 設定（solver/shift/assignment/Ωm/重み等）で RMSE がどの程度動くかを overlay metrics から集約した要約図）
- `output/cosmology/cosmology_bao_recon_gap_summary_metrics.json`（数値）
- `output/cosmology/cosmology_bao_recon_gap_broadband_fit.png`（宇宙論：残る ξ2 ギャップが broadband（a0+a1/r+a2/r²）でどこまで吸収できるかを定量化（Ross fit の周辺化に合わせた解釈の切り分け））
- `output/cosmology/cosmology_bao_recon_gap_broadband_fit_metrics.json`（数値）
- `output/cosmology/cosmology_bao_recon_gap_broadband_sensitivity.png`（宇宙論：broadband吸収の頑健性チェック（Ross bincent0..4 / s_min=30 vs 50 の感度））
- `output/cosmology/cosmology_bao_recon_gap_broadband_sensitivity_metrics.json`（数値）
- `output/cosmology/cosmology_bao_recon_gap_ross_eval_alignment.png`（宇宙論：recon gap の残差評価を Ross 一次コード（dv+cov+broadband）基準へ揃え、併せて ξ4（モデル側）がAP変換で ξ2 へ混合する大きさを定量化）
- `output/cosmology/cosmology_bao_recon_gap_ross_eval_alignment_metrics.json`（数値）
- `output/cosmology/cosmology_bao_xi_ngc_sgc_split_summary.png`（宇宙論：BAO（catalog-based ξℓ）の NGC/SGC split による内部差（Δs_peak と s²ξℓ のRMSE）を要約）
- `output/cosmology/cosmology_bao_xi_ngc_sgc_split_summary_metrics.json`（数値）
- `output/cosmology/cosmology_bao_catalog_window_multipoles.png`（宇宙論：窓関数/選択関数の異方性指標（RR2/RR0, SS2/SS0）を catalog-based の pair-count grid から可視化）
- `output/cosmology/cosmology_bao_catalog_window_multipoles_metrics.json`（数値）
- `output/cosmology/cosmology_bao_catalog_window_mixing.png`（宇宙論：窓関数の ξ0↔ξ2 window mixing の規模推定（m20 と s²(m20·ξ0)）。ξ2ギャップが mixing で説明できるかの切り分け）
- `output/cosmology/cosmology_bao_catalog_window_mixing_metrics.json`（数値）
- `output/cosmology/cosmology_bao_catalog_weight_scheme_sensitivity.png`（宇宙論：BAO一次情報（galaxy+random）の重み付け仕様（BOSS標準 / FKPのみ / 重みなし）が ε（AP warping）へ与える感度）
- `output/cosmology/cosmology_bao_catalog_weight_scheme_sensitivity_metrics.json`（数値）
- `output/cosmology/cosmology_bao_catalog_random_max_rows_sensitivity.png`（宇宙論：BAO一次情報（galaxy+random）の random 行数（max_rows）が ε（AP warping）へ与える感度）
- `output/cosmology/cosmology_bao_catalog_random_max_rows_sensitivity_metrics.json`（数値）
- `output/cosmology/cosmology_bao_pk_multipole_peakfit_window.png`（宇宙論：BAO一次統計（BOSS DR12 post-recon P(k) multipoles）からのピークfit（smooth+peak）。窓関数（RR multipoles）畳み込み込みで、AP異方（ε）をクロスチェック）
- `output/cosmology/cosmology_bao_pk_multipole_peakfit_window_metrics.json`（数値）
- `output/cosmology/cosmology_bao_pk_multipole_peakfit_pre_recon_window.png`（宇宙論：BAO一次統計（BOSS DR12 pre-recon P(k) multipoles）でのクロスチェック（smooth+peak）。窓関数畳み込み込み）
- `output/cosmology/cosmology_bao_pk_multipole_peakfit_pre_recon_window_metrics.json`（数値）
- `output/cosmology/cosmology_tolman_surface_brightness_constraints.png`（宇宙論：Tolman表面輝度の一次ソース制約（参考））
- `output/cosmology/cosmology_tolman_surface_brightness_constraints_metrics.json`（数値）
- `output/cosmology/cosmology_static_infinite_hypothesis_pack.png`（宇宙論：静的無限空間仮説（最小仮定）の検証パック（主要プローブの棄却度と前提））
- `output/cosmology/cosmology_static_infinite_hypothesis_pack_metrics.json`（数値）
- `output/cosmology/cosmology_tension_attribution.png`（宇宙論：張力の原因切り分け（距離指標依存の張力 vs 独立プローブの整合））
- `output/cosmology/cosmology_tension_attribution_metrics.json`（数値）

### JWST/MAST（スペクトル一次データ：x1d）

- `output/cosmology/jwst_spectra__jades_gs_z14_0__x1d_qc.png`（JWST/MAST x1d：スペクトルのQC（取得済みFITSの重ね描き））
- `output/cosmology/jwst_spectra__jades_gs_z14_0__z_diagnostic.png`（JWST/MAST x1d：線候補→z推定（診断図））
- `output/cosmology/jwst_spectra__jades_gs_z14_0__z_estimate.csv`（JWST/MAST x1d：z推定（候補表））
- `output/cosmology/jwst_spectra__jades_gs_z14_0__z_confirmed.png`（JWST/MAST x1d：手動線同定に基づく z確定（代表スペクトル））
- `output/cosmology/jwst_spectra__jades_gs_z14_0__z_confirmed_summary.png`（JWST/MAST x1d：z確定（spectra cross-check の要約））
- `output/cosmology/jwst_spectra__rx_11027__x1d_qc.png`（JWST/MAST x1d：スペクトルのQC（取得済みFITSの重ね描き））
- `output/cosmology/jwst_spectra__rx_11027__z_diagnostic.png`（JWST/MAST x1d：線候補→z推定（診断図））
- `output/cosmology/jwst_spectra__rx_11027__z_estimate.csv`（JWST/MAST x1d：z推定（候補表））
- `output/cosmology/jwst_spectra__rx_11027__z_confirmed_summary.png`（JWST/MAST x1d：z確定（RX_11027；spectra cross-check の要約））
- `output/cosmology/jwst_spectra__jades_10058975__x1d_qc.png`（JWST/MAST x1d：スペクトルのQC（取得済みFITSの重ね描き））
- `output/cosmology/jwst_spectra__jades_10058975__z_diagnostic.png`（JWST/MAST x1d：線候補→z推定（診断図））
- `output/cosmology/jwst_spectra__jades_10058975__z_estimate.csv`（JWST/MAST x1d：z推定（候補表））
- `output/cosmology/jwst_spectra__lid_568__z_confirmed_summary.png`（JWST/MAST x1d：z確定（LID-568；spectra cross-check の要約））
- `output/cosmology/jwst_spectra__lid2619__z_confirmed.png`（JWST/MAST x1d：z確定（LID2619；代表スペクトル））
- `output/cosmology/jwst_spectra__lid4959__z_confirmed.png`（JWST/MAST x1d：z確定（LID4959；代表スペクトル））
- `output/cosmology/jwst_spectra__glass_z12__z_confirmed_summary.png`（JWST/MAST x1d：z確定（GLASS-z12；spectra cross-check の要約））
- `output/cosmology/jwst_spectra__ceers2_5429__z_confirmed_summary.png`（JWST/MAST x1d：z確定（CEERS2-5429；spectra cross-check の要約））
