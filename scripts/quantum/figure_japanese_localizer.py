"""Part III figure text localization helper.

目的:
  Part III（量子）の図テキストを、保存直前に `ja/en` で切り替える。

前提:
  - 変換対象は Matplotlib figure 上の Text artist。
  - 数式 (`$...$`) の内部は触らない。
  - source 文字列は英語寄りのまま保持し、保存時に locale を適用する。

切替:
- `WAVEP_FIGURE_LANG=ja|en`
- `WAVEP_FIGURE_LOCALE=ja|en|...`
- 未指定時は `ja`

補足:
- `WAVEP_FIGURE_LANG` は図中テキストの表示言語を切り替える。
- `WAVEP_FIGURE_LOCALE` は artifact の保存先を切り替える。
- `ja` は現行の canonical path を維持し、非 `ja` は `locales/<locale>/...`
  へ保存して日本語図を比較基準として残す。
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Iterable


_MATH_TOKEN_RE = re.compile(r"(\$[^$]*\$)")
_JAPANESE_CHAR_RE = re.compile(r"[\u3040-\u30ff\u3400-\u9fff\uff00-\uffef]")
_LOCALE_ENV_NAMES = ("WAVEP_FIGURE_LANG", "WAVEP_MPL_FIGURE_LANG")
_TRUE_VALUES = {"1", "true", "yes", "on"}

_BASE_GLOSSARY: list[tuple[str, str]] = [
    (r"\bcoincidence window half-width\b", "コインシデンス窓半幅"),
    (r"\bwindow half-width\b", "窓半幅"),
    (r"\bstart offset\b", "開始オフセット"),
    (r"\bevent-ready\b", "event-ready"),
    (r"\bpulse separation\b", "パルス間隔"),
    (r"\btilt angle\b", "傾斜角"),
    (r"\bphase shift\b", "位相差"),
    (r"\bphase scaling\b", "位相スケーリング"),
    (r"\bphase\b", "位相"),
    (r"\bgravimeter\b", "重力計"),
    (r"\batom interferometer\b", "原子干渉計"),
    (r"\boptical clock\b", "光格子時計"),
    (r"\bchronometric leveling\b", "クロノメトリック測地"),
    (r"\bgravity-induced\b", "重力誘起"),
    (r"\bdephasing\b", "位相緩和"),
    (r"\bdecoherence\b", "デコヒーレンス"),
    (r"\binterrogation time\b", "観測時間"),
    (r"\brequired\b", "必要"),
    (r"\bvisibility\b", "可視度"),
    (r"\bcontrast\b", "コントラスト"),
    (r"\bnoise budget\b", "ノイズ予算"),
    (r"\bnoise\b", "ノイズ"),
    (r"\brun-to-run\b", "run-to-run"),
    (r"\bsingle-photon\b", "単一光子"),
    (r"\bphoton gas\b", "光子気体"),
    (r"\bphoton density\b", "光子数密度"),
    (r"\bphoton flux\b", "光子フラックス"),
    (r"\bmatter wave\b", "物質波"),
    (r"\binterference\b", "干渉"),
    (r"\bprecision\b", "精度"),
    (r"\baudit\b", "監査"),
    (r"\bunified\b", "統合"),
    (r"\bconsistency\b", "整合性"),
    (r"\bcross-check\b", "クロスチェック"),
    (r"\bbaseline\b", "基準"),
    (r"\bsummary\b", "要約"),
    (r"\bsensitivity\b", "感度"),
    (r"\bselection\b", "選別"),
    (r"\bsweep\b", "掃引"),
    (r"\bwindow\b", "窓幅"),
    (r"\bpairs\b", "ペア数"),
    (r"\bpair\b", "ペア"),
    (r"\bcount\b", "件数"),
    (r"\bcounts\b", "件数"),
    (r"\bdelay\b", "遅延"),
    (r"\boffset\b", "オフセット"),
    (r"\bcovariance\b", "共分散"),
    (r"\bcorrelation\b", "相関"),
    (r"\beigenvalue\b", "固有値"),
    (r"\beigenvalues\b", "固有値"),
    (r"\bsystematics\b", "系統誤差"),
    (r"\bbudget\b", "予算"),
    (r"\bdecomposition\b", "分解"),
    (r"\blongterm\b", "長期"),
    (r"\bfalsification pack\b", "反証パック"),
    (r"\boperational I/F\b", "運用 I/F"),
    (r"\boperational\b", "運用"),
    (r"\bproxy\b", "proxy"),
    (r"\bdirect\b", "直接"),
    (r"\bglobal\b", "全体"),
    (r"\blocal\b", "局所"),
    (r"\bdifferential\b", "差分"),
    (r"\bquantification\b", "定量"),
    (r"\bpredictions\b", "予測"),
    (r"\bprediction\b", "予測"),
    (r"\bdistribution\b", "分布"),
    (r"\btargets\b", "目標"),
    (r"\btarget\b", "目標"),
    (r"\bobserved\b", "観測"),
    (r"\bpredicted\b", "予測"),
    (r"\bobservation\b", "観測"),
    (r"\bresidual\b", "残差"),
    (r"\bresiduals\b", "残差"),
    (r"\bmedian\b", "中央値"),
    (r"\btop-10 nuclei\b", "上位10核種"),
    (r"\brepresentative nuclei\b", "代表核種"),
    (r"\blight nuclei\b", "軽核"),
    (r"\ball nuclei\b", "全核種"),
    (r"\bnuclear force\b", "核力"),
    (r"\bnear-field\b", "近接場"),
    (r"\btwo-mode\b", "二モード"),
    (r"\bbinding energy\b", "束縛エネルギー"),
    (r"\bbinding scale\b", "束縛スケール"),
    (r"\bbound-state\b", "束縛状態"),
    (r"\bbound state\b", "束縛状態"),
    (r"\bbound condition\b", "束縛条件"),
    (r"\bfrequency mapping\b", "周波数写像"),
    (r"\bverification\b", "検証"),
    (r"\bminimal additional physics\b", "最小追加物理"),
    (r"\btheory-difference\b", "理論差分"),
    (r"\btheory diff\b", "理論差分"),
    (r"\bprecision requirement\b", "精度要求"),
    (r"\bHe-4 binding\b", "He-4 束縛"),
    (r"\bDeuteron binding\b", "重水素束縛"),
    (r"\bdeuteron\b", "重水素"),
    (r"\bsinglet\b", "一重項"),
    (r"\btriplet\b", "三重項"),
    (r"\bscattering length\b", "散乱長"),
    (r"\beffective range\b", "有効レンジ"),
    (r"\blow-energy parameters\b", "低エネルギー・パラメータ"),
    (r"\bsquare-well\b", "角型井戸"),
    (r"\bsquare well\b", "角型井戸"),
    (r"\bstanding-wave\b", "定常波"),
    (r"\bwell range\b", "井戸レンジ"),
    (r"\brequired depth\b", "必要深さ"),
    (r"\billustration\b", "模式図"),
    (r"\bSilicon\b", "シリコン"),
    (r"\bCopper\b", "銅"),
    (r"\bthermal expansion\b", "熱膨張"),
    (r"\bheat capacity\b", "熱容量"),
    (r"\bbulk modulus\b", "体積弾性率"),
    (r"\bresistivity\b", "抵抗率"),
    (r"\btemperature coefficient\b", "温度係数"),
    (r"\blattice\b", "格子"),
    (r"\bholdout\b", "ホールドアウト"),
    (r"\btemperature-split\b", "温度分割"),
    (r"\bphonon\b", "フォノン"),
    (r"\banchor\b", "アンカー"),
    (r"\bblackbody radiation\b", "黒体放射"),
    (r"\bblackbody\b", "黒体"),
    (r"\bentropy density\b", "エントロピー密度"),
    (r"\benergy density\b", "エネルギー密度"),
    (r"\bentropy per photon\b", "光子あたりエントロピー"),
    (r"\bentropy\b", "エントロピー"),
    (r"\btemperature\b", "温度"),
    (r"\bvacuum wavelength\b", "真空波長"),
    (r"\bmeasured / predicted\b", "観測 / 予測"),
    (r"\bmeasured\b", "観測"),
    (r"\bpred\b", "予測"),
    (r"\bHydrogen\b", "水素"),
    (r"\bHelium\b", "ヘリウム"),
    (r"\bisotopic reduced-mass scaling\b", "同位体換算質量スケーリング"),
    (r"\bdissociation\b", "解離"),
    (r"\bspectroscopic\b", "分光"),
    (r"\bQED vacuum\b", "QED 真空"),
    (r"\bVacuum \+ QED precision observables\b", "真空 + QED 精密観測量"),
    (r"\bCasimir\b", "Casimir"),
    (r"\bLamb shift\b", "Lamb シフト"),
    (r"\bscaling\b", "スケーリング"),
    (r"\bforce scale\b", "力スケール"),
    (r"\brelative scale\b", "相対スケール"),
    (r"\bnuclear-size term\b", "核サイズ項"),
    (r"\bsource coverage by record type\b", "記録種別ごとの出典カバー率"),
    (r"\bobserved vs predicted\b", "観測と予測"),
    (r"\brepresentative\b", "代表"),
    (r"\bavailable rows\b", "利用可能行"),
    (r"\bheight-equivalent\b", "高さ換算"),
    (r"\bgeopotential difference\b", "重力ポテンシャル差"),
    (r"\bcurrent\b", "現在"),
]

_STEM_SPECIFIC_GLOSSARY: dict[str, list[tuple[str, str]]] = {
    "bell_selection_sensitivity_summary": [
        (r"Weihs 1998: \|S\| vs window \([^)]+\)", "Weihs 1998: |S| と窓幅"),
        (r"\|S\| vs window", "|S| と窓幅"),
        (r"NIST: CH J_prob vs window\s*\|\s*KS\(A\)=[^|]+\|\s*KS\(B\)=[^\n]+", "NIST: CH J_prob と窓幅"),
        (r"NIST: CH J_prob vs window", "NIST: CH J_prob と窓幅"),
        (r"Delft \(event-ready\): CHSH S vs start offset", "Delft: CHSH S と開始オフセット"),
        (r"CHSH S vs start offset", "CHSH S と開始オフセット"),
        (r"Bell selection sensitivity summary", "Bell 選別感度要約"),
    ],
    "cow_phase_shift": [
        (r"COW \(gravity-induced quantum interference\): phase scaling", "COW（重力誘起量子干渉）: 位相差スケーリング"),
        (r"COW phase scaling vs tilt angle", "COW 位相スケーリングと傾斜角"),
        (r"H-v sweep:", "H-v 掃引:"),
    ],
    "optical_clock_chronometric_leveling": [
        (r"Optical clock chronometric leveling", "光格子時計クロノメトリック測地"),
    ],
    "gravity_quantum_interference_delta_predictions": [
        (r"atom-interferometer unified audit", "原子干渉計統合監査"),
        (r"P-model vs GR; Earth field", "P-model と GR; 地球重力場"),
    ],
    "matter_wave_interference_precision_audit": [
        (r"matter-wave interference precision audit", "物質波干渉精度監査"),
    ],
    "gravity_induced_decoherence": [
        (r"gravity-induced dephasing", "重力誘起位相緩和"),
        (r"time-structure", "時間構造"),
    ],
    "hom_squeezed_light_unified_audit": [
        (r"HOM squeezed-light unified audit", "HOM スクイーズド光統合監査"),
    ],
    "qed_vacuum_precision": [
        (r"Vacuum \+ QED precision observables(?:\s*\([^)]+\))?", "真空 + QED 精密観測量"),
        (r"Casimir: sphere–plate force scale", "Casimir: 球-平板スケール"),
        (r"Lamb shift: scaling\s*\(why Z>1 helps\)", "Lamb シフト: Z 依存スケール"),
        (r"nuclear-size term\s*\(example; Table 4\)", "核サイズ項（例; Table 4）"),
        (r"alpha precision cross-check", "α 精度クロスチェック"),
    ],
    "atomic_hydrogen_baseline": [
        (r"Atomic baseline \(Hydrogen, NIST ASD\)", "原子基準（水素, NIST ASD）"),
    ],
    "atomic_helium_baseline": [
        (r"Atomic baseline \(Helium, NIST ASD\)", "原子基準（ヘリウム, NIST ASD）"),
    ],
    "molecular_isotopic_scaling": [
        (r"isotopic reduced-mass scaling", "同位体換算質量スケーリング"),
    ],
    "thermo_blackbody_radiation_baseline": [
        (r"Thermo baseline", "熱力学基準"),
    ],
    "thermo_blackbody_entropy_baseline": [
        (r"Thermo baseline", "熱力学基準"),
        (r"second law", "第二法則"),
    ],
    "nuclear_binding_energy_frequency_mapping_differential_quantification": [
        (r"quantitative differential predictions and precision targets", "定量的差分予測と精度目標"),
    ],
    "nuclear_binding_energy_frequency_mapping_minimal_additional_physics": [
        (r"frozen falsification threshold", "固定反証閾値"),
    ],
    "nuclear_binding_energy_frequency_mapping_theory_diff": [
        (r"theory-difference extraction", "理論差分抽出"),
    ],
    "nuclear_near_field_interference_two_mode_model": [
        (r"nuclear force as near-field interference", "核力の近接場干渉"),
    ],
    "nuclear_binding_energy_frequency_mapping_deuteron_two_body": [
        (r"deuteron \(pn\) two-body: bound-state scales \(frozen\)", "重水素（pn）二体: 束縛状態スケール（固定）"),
        (r"Square-well example \(s-wave\):", "角型井戸の例（s波）:"),
        (
            r"This is an operational I/F for the standing-wave \(bound\) condition;",
            "これは定常波（束縛）条件の運用 I/F を示すものであり、",
        ),
        (
            r"it does not claim the nuclear force is literally a square well\.",
            "核力そのものが文字通りの角型井戸だと主張するものではない。",
        ),
        (r"Square-well depth required to support B \(illustration\)", "B を支えるために必要な角型井戸深さ（模式図）"),
        (r"well range R \(fm\)", "井戸レンジ R (fm)"),
        (r"required depth V0 \(MeV\)", "必要深さ V0 (MeV)"),
        (r"deuteron Δω mapping via 2-body boundary condition", "二体境界条件による重水素 Δω 写像"),
    ],
    "quantum_scoreboard": [
        (r"Quantum scoreboard \(green=OK / yellow=needs work / red=mismatch\)", "総合スコアボード（量子：緑=OK / 黄=要改善 / 赤=不一致）"),
        (r"Bell \(public primary\)", "ベル（公開一次）"),
        (r"Gravity×quantum interference: COW", "重力×量子干渉：COW"),
        (r"Nuclear \(wave interference \+ deuteron \+ np scattering\)", "原子核（波動干渉＋deuteron+np散乱）"),
        (r"Gravity-induced decoherence", "重力誘起デコヒーレンス"),
        (r"Photon quantum interference", "光の量子干渉"),
        (r"Atomic/molecular baselines", "原子・分子（基準値）"),
    ],
    "bell_selection_sensitivity_summary": [
        (r"\|S\| \(fixed variant\)", "|S|（固定 variant）"),
        (r"\|S\| \(fixed variant\)", "|S|(固定 variant)"),
        (r"raw series", "原系列"),
        (r"accidental corrected", "accidental 補正後"),
        (r"recommended window ≈ 22.3 ns", "推奨窓 ≈ 22.3 ns"),
        (r"coincidence-based J_prob", "coincidence-base J_prob"),
        (r"trial-based J_prob", "trial-base J_prob"),
    ],
    "falsification_pack": [
        (r"delay setting dependence \(Δmedian; z\)", "遅延の setting 依存（Δmedian; z）"),
        (r"delay setting dependence \(Δmedian; z\)", "delay setting 依存(Δmedian; z)"),
    ],
    "nist_belltest_time_tag_bias__03_43_afterfixingModeLocking_s3600": [
        (
            r"NIST Bell test \(time-tag\): setting-dependent delay and coincidence-window sensitivity",
            "NIST Bell test（time-tag）：setting 依存遅延と coincidence 窓感度",
        ),
        (r"window dependence \(GPS PPS alignment\)", "窓幅依存（GPS PPS 整列）"),
        (r"NIST Bell test \(time-tag\): setting-dependent delay and coincidence-window sensitivity", "NIST Bell test(time-tag): setting 依存delay and coincidence 窓sensitivity"),
        (r"window dependence \(GPS PPS alignment\)", "window依存(GPS PPS 整列)"),
        (r"greedy pair count", "greedy pair 数"),
    ],
    "nist_belltest_trial_based__03_43_afterfixingModeLocking_s3600": [
        (r"NIST Bell test: trial-based aggregation and differential", "NIST Bell test：trial-base 集計との差分"),
        (r"coincidence-based and trial-based consistency", "coincidence-base と trial-base の一致"),
        (r"CH J_prob\(A1=0,B1=0\) window dependence", "CH J_prob（A1=0,B1=0）の窓幅依存"),
        (r"NIST Bell test: trial-based aggregation and differential", "NIST Bell test: trial-base 集計 and differential"),
        (r"coincidence-based and trial-based consistency", "coincidence-base and trial-base 一致"),
        (r"CH J_prob\(A1=0,B1=0\) window dependence", "CH J_prob(A1=0,B1=0) window依存"),
        (r"nearest", "最接近"),
    ],
    "electron_double_slit_interference": [
        (r"electron double-slit diffraction \(600 eV; 50 nm slit, 280 nm sep\.\)", "電子二重スリット回折（600 eV；50 nm slit，280 nm sep.）"),
        (r"P12 \(double slit; coherent\)", "P12（二重スリット；coherent）"),
        (r"P1\+P2 \(sum of single slits\)", "P1+P2（単スリット和）"),
        (r"P1 \(single-slit envelope\)", "P1（単スリット包絡）"),
        (r"electron double-slit diffraction \(600 eV; 50 nm slit, 280 nm sep\.\)", "電子二重スリット回折(600 eV；50 nm slit, 280 nm sep.)"),
        (r"P12 \(double slit; coherent\)", "P12(二重スリット；coherent)"),
        (r"P1\+P2 \(sum of single slits\)", "P1+P2(単スリット和)"),
        (r"P1 \(single-slit envelope\)", "P1(単スリット包絡)"),
        (r"normalized intensity \(arb\.\)", "規格化強度 (arb.)"),
        (r"diffraction angle θ \(mrad\)", "回折角 θ (mrad)"),
    ],
    "matter_wave_interference_precision_audit": [
        (r"matter-wave interference precision audit", "物質波干渉の精度監査"),
        (r"electron double-slit angular scale", "電子二重スリットの角スケール"),
        (r"cross-channel consistency \(\|z\|\)", "クロスチャネル整合（|z|）"),
        (r"atomic α consistency", "原子 α 整合"),
        (r"atom interferometer precision gap", "原子干渉計の精度ギャップ"),
        (r"molecular isotopic scaling residual", "分子の同位体スケーリング残差"),
        (r"electron double-slit angular scale", "電子二重スリット Angular scale"),
        (r"cross-channel consistency \(\|z\|\)", "クロスチャネルconsistency(|z|)"),
        (r"atomic α consistency", "原子 α 整合"),
        (r"Atoms", "原子"),
        (r"atom interferometer precision gap", "atom interferometer precisionギャップ"),
        (r"molecular isotopic scaling residual", "分子 同位体scalingresidual"),
        (r"Molecules", "分子"),
    ],
    "gravity_induced_decoherence": [
        (r"gravity-induced decoherence: observed quantity and noise budget", "重力誘起デコヒーレンス：観測量とノイズ予算"),
        (
            r"gravity-induced dephasing \(optical clock ensemble; Gaussian height distribution\)",
            "重力誘起位相緩和（光格子時計 ensemble；Gaussian 高さ分布）",
        ),
        (r"required σ_y \(RMS fractional frequency noise\)", "必要な σ_y（RMS相対周波数雑音）"),
        (r"P-model time-structure: σ_y that mimics decoherence", "P-model 時間構造：decoherence を模擬する σ_y"),
        (r"gravity-induced decoherence: observed quantity and noise budget", "gravity-induceddecoherence: observed量 and noise budget"),
        (r"gravity-induced dephasing \(optical clock ensemble; Gaussian height distribution\)", "gravity-induced dephasing(optical clock ensemble；Gaussian 高さdistribution)"),
        (r"required σ_y \(RMS fractional frequency noise\)", "requiredな σ_y(RMS相対周波数雑音)"),
        (r"P-model time-structure: σ_y that mimics decoherence", "P-model time-structure: decoherence 模擬する σ_y"),
    ],
    "photon_quantum_interference": [
        (r"visibility V", "可視度 V"),
        (r"equivalent path-length noise σL \(nm\)", "等価な経路長ノイズ σL (nm)"),
        (r"single-photon interference: V → σL", "単一光子干渉：V → σL"),
        (r"reported \(QD2; corrected\)", "reported（QD2; corrected）"),
        (r"separation time D \(ns\)", "分離時間 D (ns)"),
        (r"HOM visibility \(%\)", "HOM 可視度 (%)"),
        (r"HOM: visibility vs separation time", "HOM：分離時間に対する可視度"),
        (
            r"definition: V=1−\(C∥/C⊥\) at zero delay \(Methods, arXiv:2106\.03871v2\)",
            "定義: V=1−(C∥/C⊥) at zero delay\n（Methods, arXiv:2106.03871v2）",
        ),
        (r"Zenodo 6371310 \(ExData Fig\.3b\)", "Zenodo 6371310（ExData Fig.3b）"),
        (r"low-frequency noise PSD \(indistinguishability proxy\)", "低周波ノイズ PSD（識別不能性の proxy）"),
        (
            r"squeezing: 10\.0 dB → variance ratio=0\.100 loss-only bound: η ≥ 0\.900",
            "スクイージング: 10.0 dB → 分散比=0.100\nloss-only bound: η ≥ 0.900",
        ),
        (r"photon interference observables \(visibility, HOM, squeezing/noise\)", "光子干渉の観測量（可視度・HOM・スクイージング/ノイズ）"),
        (r"equivalent path-length noise σL \(nm\)", "等価な経路長noise σL (nm)"),
        (r"photon interference observables \(visibility, HOM, squeezing/noise\)", "光子interference observed量(visibility・HOM・スクイージング/noise)"),
        (r"HOM: visibility vs separation time", "HOM: 分離時間に対するvisibility"),
        (r"definition: V=1−\(C∥/C⊥\) at zero delay \(Methods, arXiv:2106\.03871v2\)", "定義: V=1−(C∥/C⊥) at zero delay (Methods, arXiv:2106.03871v2)"),
        (r"separation time D \(ns\)", "分離時間 D (ns)"),
        (r"low-frequency noise PSD \(indistinguishability proxy\)", "低周波noise PSD(識別不能性 proxy)"),
        (r"squeezing: 10\.0 dB → variance ratio=0\.100 loss-only bound: η ≥ 0\.900", "スクイージング: 10.0 dB → 分散比=0.100 loss-only bound: η ≥ 0.900"),
        (r"frequency \(Hz\)", "周波数 (Hz)"),
    ],
    "hom_squeezed_light_unified_audit": [
        (r"HOM \+ squeezed-light unified audit", "HOM + スクイーズド光の統合監査"),
        (r"HOM visibility \(50% baseline included\)", "HOM 可視度（50% 基準付き）"),
        (r"HOM significance", "HOM の有意性"),
        (r"squeezing scale", "スクイージング規模"),
        (r"noise PSD scale indicator", "ノイズ PSD の規模指標"),
        (r"HOM \+ squeezed-light unified audit", "HOM + スクイーズド光 unifiedaudit"),
        (r"HOM visibility \(50% baseline included\)", "HOM visibility(50% baseline付き)"),
        (r"HOM significance", "HOM 有意性"),
        (r"squeezing scale", "スクイージング規模"),
        (r"variance ratio", "分散比"),
        (r"noise PSD scale indicator", "noise PSD 規模指標"),
    ],
    "qed_vacuum_precision": [
        (r"Vacuum \+ QED precision observables", "真空 + QED 精密観測量"),
        (r"Casimir: sphere-plane scale", "Casimir：球-平板スケール"),
        (r"ideal conductor \(PFA\)", "理想導体（PFA）"),
        (r"separation a \(nm\)", "間隔 a (nm)"),
        (r"Lamb shift: Z-dependent scale", "Lamb シフト：Z 依存スケール"),
        (r"relative scale \(Z=1 → 1\)", "相対スケール（Z=1 → 1）"),
        (r"source: physics/0009069v1 \(scaling discussion\)", "出典: physics/0009069v1（スケーリング記述）"),
        (r"nuclear-size term \(example; Table 4\)", "核サイズ項（例; Table 4）"),
        (r"non-QED systematic \(radius\) example", "非QED系統（半径）の例"),
        (r"atomic-gravity order of magnitude", "原子重力のオーダー"),
        (r"recoil \(Rb; 0812\.3139\)", "反跳（Rb; 0812.3139）"),
        (r"required epsilon", "必要な epsilon（recoil）"),
        (r"Casimir: sphere-plane scale", "Casimir: 球-平板スケール"),
        (r"reported precision ≈ 1% \(smallest a\)", "報告精度 ≈ 1%（最小 a）"),
        (r"ideal conductor \(PFA\)", "理想導体（PFA）"),
        (r"separation a \(nm\)", "間隔 a (nm)"),
        (r"Lamb shift: Z-dependent scale", "Lamb シフト：Z 依存スケール"),
        (r"unknown 2-loop ∝ Z\^6", "未知 2-loop ∝ Z^6"),
        (r"source: physics/0009069v1 \(scaling discussion\)", "出典: physics/0009069v1(scaling記述)"),
        (r"non-QED systematic \(radius\) example", "非QEDsystematic(半径) 例"),
        (r"atomic-gravity order of magnitude", "原子重力 オーダー"),
        (r"required epsilon", "requiredな epsilon"),
    ],
    "nuclear_binding_deuteron": [
        (r"deuteron nuclear baseline \(observed quantities fixed\)", "重水素の核ベースライン（観測量固定）"),
        (r"mass-defect baseline \(CODATA/NIST\)", "質量欠損ベースライン（CODATA/NIST）"),
        (r"ledger value", "記帳値"),
        (r"size constraints \(radius and binding tail\)", "サイズ制約（半径と束縛テール）"),
        (r"1/κ from B \(tail scale\)", "Bから得る 1/κ（テール尺度）"),
        (r"r_d \(charge rms radius\)", "r_d（電荷rms半径）"),
        (r"deuteron nuclear baseline \(observed quantities fixed\)", "重水素 核ベースライン(Observed量固定)"),
        (r"mass-defect baseline \(CODATA/NIST\)", "質量欠損ベースライン(CODATA/NIST)"),
        (r"ledger value", "記帳value"),
        (r"size constraints \(radius and binding tail\)", "サイズ制約(半径 and 束縛テール)"),
        (r"1/κ from B \(tail scale\)", "Bから得る 1/κ(テール尺度)"),
        (r"r_d \(charge rms radius\)", "r_d(電荷rms半径)"),
        (r"length scale \(fm\)", "長さスケール (fm)"),
    ],
    "nuclear_np_scattering_baseline": [
        (r"np scattering baseline \(observed quantities fixed\)", "np scattering baseline（観測量固定）"),
        (r"np scattering baseline \(observed quantities fixed\)", "np scattering baseline(observed量固定)"),
    ],
    "nuclear_effective_potential_two_range_fit_as_rs_eq18": [
        (r"two-range hypothesis: fit triplet\(B,a_t,r_t,v2t\) and singlet\(a_s,r_s\), then predict v2s", "2レンジ仮説: 三重項(B,a_t,r_t,v2t)と一重項(a_s,r_s)をフィットし、v2sを予測"),
        (r"eq18 \(GWU/SAID\) two-range well \(shared geometry\)", "eq18（GWU/SAID）\n2レンジ井戸（幾何共有）"),
        (r"Potential V\(r\) \(MeV\)", "ポテンシャル V(r)（MeV）"),
        (r"triplet \(fit B, a_t, r_t, v2t\)", "三重項（B, a_t, r_t, v2t をフィット）"),
        (r"singlet \(fit V1, V2 by a_s, r_s\)", "一重項（a_s, r_sで V1, V2 をフィット）"),
        (r"triplet: ERE fit \(v2 targets\)", "三重項: EREフィット（v2を目標）"),
        (r"k grid", "kグリッド"),
        (r"ERE fit", "EREフィット"),
        (r"fit/predictions − observed \(units: fm³, fm, fm³\)", "フィット/予測 − 観測（単位: fm³, fm, fm³）"),
        (r"effective-range function kcot δ \(fm−1\)", "有効レンジ関数 kcot δ（fm−1）"),
        (r"two-range hypothesis: fit triplet\(B,a_t,r_t,v2t\) and singlet\(a_s,r_s\), then predict v2s", "2レンジ仮説: triplet(B,a_t,r_t,v2t) and singlet(a_s,r_s) フィットし, v2s predictions"),
        (r"eq18 \(GWU/SAID\) two-range well \(shared geometry\)", "eq18 (GWU/SAID) 2レンジ井戸(幾何共有)"),
        (r"Potential V\(r\) \(MeV\)", "ポテンシャル V(r)(MeV)"),
        (r"triplet\(B, a_t, r_t, v2t fit\)", "triplet(B, a_t, r_t, v2t フィット)"),
        (r"singlet\(a_s, r_s with V1, V2 fit\)", "singlet(a_s, r_sで V1, V2 フィット)"),
        (r"triplet: ERE fit \(v2 targets\)", "triplet: EREフィット(v2 targets)"),
        (r"k grid", "kグリッド"),
        (r"ERE fit", "EREフィット"),
        (r"fit/predictions − observed \(units: fm³, fm, fm³\)", "フィット/predictions − observed(単位: fm³, fm, fm³)"),
        (r"effective-range function kcot δ \(fm−1\)", "effective range関数kcot δ(fm−1)"),
    ],
    "nuclear_effective_potential_two_range_fit_as_rs_eq19": [
        (r"two-range hypothesis: fit triplet\(B,a_t,r_t,v2t\) and singlet\(a_s,r_s\), then predict v2s", "2レンジ仮説: 三重項(B,a_t,r_t,v2t)と一重項(a_s,r_s)をフィットし、v2sを予測"),
        (r"eq19 \(Nijmegen\) two-range well \(shared geometry\)", "eq19（Nijmegen）\n2レンジ井戸（幾何共有）"),
        (r"Potential V\(r\) \(MeV\)", "ポテンシャル V(r)（MeV）"),
        (r"triplet \(fit B, a_t, r_t, v2t\)", "三重項（B, a_t, r_t, v2t をフィット）"),
        (r"singlet \(fit V1, V2 by a_s, r_s\)", "一重項（a_s, r_sで V1, V2 をフィット）"),
        (r"triplet: ERE fit \(v2 targets\)", "三重項: EREフィット（v2を目標）"),
        (r"k grid", "kグリッド"),
        (r"ERE fit", "EREフィット"),
        (r"fit/predictions − observed \(units: fm³, fm, fm³\)", "フィット/予測 − 観測（単位: fm³, fm, fm³）"),
        (r"effective-range function kcot δ \(fm−1\)", "有効レンジ関数 kcot δ（fm−1）"),
        (r"two-range hypothesis: fit triplet\(B,a_t,r_t,v2t\) and singlet\(a_s,r_s\), then predict v2s", "2レンジ仮説: triplet(B,a_t,r_t,v2t) and singlet(a_s,r_s) フィットし, v2s predictions"),
        (r"eq19 \(Nijmegen\) two-range well \(shared geometry\)", "eq19 (Nijmegen) 2レンジ井戸(幾何共有)"),
        (r"Potential V\(r\) \(MeV\)", "ポテンシャル V(r)(MeV)"),
        (r"triplet\(B, a_t, r_t, v2t fit\)", "triplet(B, a_t, r_t, v2t フィット)"),
        (r"singlet\(a_s, r_s with V1, V2 fit\)", "singlet(a_s, r_sで V1, V2 フィット)"),
        (r"triplet: ERE fit \(v2 targets\)", "triplet: EREフィット(v2 targets)"),
        (r"k grid", "kグリッド"),
        (r"ERE fit", "EREフィット"),
        (r"fit/predictions − observed \(units: fm³, fm, fm³\)", "フィット/predictions − observed(単位: fm³, fm, fm³)"),
        (r"effective-range function kcot δ \(fm−1\)", "effective range関数kcot δ(fm−1)"),
    ],
    "nuclear_effective_potential_pion_constrained_barrier_tail_kq_scan_eq18": [
        (
            r"λπ-constrained three-range \(barrier\+tail split, free tail depth\) \+ \(k,q\) scan",
            "λπ拘束3レンジ（障壁+テール分割、テール深さ自由）+ (k,q)走査",
        ),
        (
            r"eq18 \(GWU/SAID\) three-range \(λπ constrained, barrier\+tail split, free tail depth\)",
            "eq18（GWU/SAID）\n3レンジ（λπ拘束、障壁+テール分割、テール深さ自由）",
        ),
        (r"Potential V\(r\) \(MeV\)", "ポテンシャル V(r)（MeV）"),
        (r"triplet \(fit B, a_t, r_t, v2t\)", "三重項（B, a_t, r_t, v2t をフィット）"),
        (r"singlet \(fit V1, V2>=0 by a_s, r_s\)", "一重項（a_s, r_sで V1, V2>=0 をフィット）"),
        (r"triplet: ERE fit \(v2 targets\)", "三重項: EREフィット（v2を目標）"),
        (r"k grid", "kグリッド"),
        (r"ERE fit", "EREフィット"),
        (r"fit/predictions − observed \(units: fm³, fm, fm³\)", "フィット/予測 − 観測（単位: fm³, fm, fm³）"),
        (r"effective-range function kcot δ \(fm−1\)", "有効レンジ関数 kcot δ（fm−1）"),
        (r"λπ-constrained three-range \(barrier\+tail split, free tail depth\) \+ \(k,q\) scan", "λπ拘束3レンジ(障壁+テール分割, テール深さ自由)+ (k,q)走査"),
        (r"eq18 \(GWU/SAID\) three-range \(λπ constrained, barrier\+tail split, free tail depth\)", "eq18 (GWU/SAID) 3レンジ(λπ拘束, 障壁+テール分割, テール深さ自由)"),
        (r"Potential V\(r\) \(MeV\)", "ポテンシャル V(r)(MeV)"),
        (r"triplet\(B, a_t, r_t, v2t fit\)", "triplet(B, a_t, r_t, v2t フィット)"),
        (r"singlet\(a_s, r_s with V1, V2>=0 fit\)", "singlet(a_s, r_sで V1, V2>=0 フィット)"),
        (r"triplet: ERE fit \(v2 targets\)", "triplet: EREフィット(v2 targets)"),
        (r"k grid", "kグリッド"),
        (r"ERE fit", "EREフィット"),
        (r"fit/predictions − observed \(units: fm³, fm, fm³\)", "フィット/predictions − observed(単位: fm³, fm, fm³)"),
        (r"effective-range function kcot δ \(fm−1\)", "effective range関数kcot δ(fm−1)"),
    ],
    "nuclear_effective_potential_pion_constrained_barrier_tail_kq_scan_eq19": [
        (
            r"λπ-constrained three-range \(barrier\+tail split, free tail depth\) \+ \(k,q\) scan",
            "λπ拘束3レンジ（障壁+テール分割、テール深さ自由）+ (k,q)走査",
        ),
        (
            r"eq19 \(Nijmegen\) three-range \(λπ constrained, barrier\+tail split, free tail depth\)",
            "eq19（Nijmegen）\n3レンジ（λπ拘束、障壁+テール分割、テール深さ自由）",
        ),
        (r"Potential V\(r\) \(MeV\)", "ポテンシャル V(r)（MeV）"),
        (r"triplet \(fit B, a_t, r_t, v2t\)", "三重項（B, a_t, r_t, v2t をフィット）"),
        (r"singlet \(fit V1, V2>=0 by a_s, r_s\)", "一重項（a_s, r_sで V1, V2>=0 をフィット）"),
        (r"triplet: ERE fit \(v2 targets\)", "三重項: EREフィット（v2を目標）"),
        (r"k grid", "kグリッド"),
        (r"ERE fit", "EREフィット"),
        (r"fit/predictions − observed \(units: fm³, fm, fm³\)", "フィット/予測 − 観測（単位: fm³, fm, fm³）"),
        (r"effective-range function kcot δ \(fm−1\)", "有効レンジ関数 kcot δ（fm−1）"),
        (r"λπ-constrained three-range \(barrier\+tail split, free tail depth\) \+ \(k,q\) scan", "λπ拘束3レンジ(障壁+テール分割, テール深さ自由)+ (k,q)走査"),
        (r"eq19 \(Nijmegen\) three-range \(λπ constrained, barrier\+tail split, free tail depth\)", "eq19 (Nijmegen) 3レンジ(λπ拘束, 障壁+テール分割, テール深さ自由)"),
        (r"Potential V\(r\) \(MeV\)", "ポテンシャル V(r)(MeV)"),
        (r"triplet\(B, a_t, r_t, v2t fit\)", "triplet(B, a_t, r_t, v2t フィット)"),
        (r"singlet\(a_s, r_s with V1, V2>=0 fit\)", "singlet(a_s, r_sで V1, V2>=0 フィット)"),
        (r"triplet: ERE fit \(v2 targets\)", "triplet: EREフィット(v2 targets)"),
        (r"k grid", "kグリッド"),
        (r"ERE fit", "EREフィット"),
        (r"fit/predictions − observed \(units: fm³, fm, fm³\)", "フィット/predictions − observed(単位: fm³, fm, fm³)"),
        (r"effective-range function kcot δ \(fm−1\)", "effective range関数kcot δ(fm−1)"),
    ],
    "nuclear_effective_potential_pion_constrained_barrier_tail_channel_split_kq_scan_triplet_barrier_fraction_scan_eq18": [
        (
            r"λπ-constrained three-range \+ channel split \(k,q\) \+ triplet barrier-fraction scan",
            "λπ拘束3レンジ + チャネル分離(k,q) + 三重項障壁比率走査",
        ),
        (
            r"eq18 \(GWU/SAID\) three-range \(triplet barrier-fraction scan\)",
            "eq18（GWU/SAID）\n3レンジ（三重項障壁比率走査）",
        ),
        (r"Potential V\(r\) \(MeV\)", "ポテンシャル V(r)（MeV）"),
        (r"triplet \(fit B, a_t, r_t, v2t\)", "三重項（B, a_t, r_t, v2t をフィット）"),
        (r"singlet \(fit V1, V2>=0 by a_s, r_s\)", "一重項（a_s, r_sで V1, V2>=0 をフィット）"),
        (r"triplet: ERE fit \(v2 targets\)", "三重項: EREフィット（v2を目標）"),
        (r"k grid", "kグリッド"),
        (r"ERE fit", "EREフィット"),
        (r"fit/predictions − observed \(units: fm³, fm, fm³\)", "フィット/予測 − 観測（単位: fm³, fm, fm³）"),
        (r"effective-range function kcot δ \(fm−1\)", "有効レンジ関数 kcot δ（fm−1）"),
        (r"λπ-constrained three-range \+ channel split \(k,q\) \+ triplet barrier-fraction scan", "λπ拘束3レンジ + チャネル分離(k,q) + triplet障壁比率走査"),
        (r"eq18 \(GWU/SAID\) three-range \(triplet barrier-fraction scan\)", "eq18 (GWU/SAID) 3レンジ(triplet障壁比率走査)"),
        (r"Potential V\(r\) \(MeV\)", "ポテンシャル V(r)(MeV)"),
        (r"triplet\(B, a_t, r_t, v2t fit\)", "triplet(B, a_t, r_t, v2t フィット)"),
        (r"singlet\(a_s, r_s with V1, V2>=0 fit\)", "singlet(a_s, r_sで V1, V2>=0 フィット)"),
        (r"triplet: ERE fit \(v2 targets\)", "triplet: EREフィット(v2 targets)"),
        (r"k grid", "kグリッド"),
        (r"ERE fit", "EREフィット"),
        (r"fit/predictions − observed \(units: fm³, fm, fm³\)", "フィット/predictions − observed(単位: fm³, fm, fm³)"),
        (r"effective-range function kcot δ \(fm−1\)", "effective range関数kcot δ(fm−1)"),
    ],
    "nuclear_effective_potential_pion_constrained_barrier_tail_channel_split_kq_scan_triplet_barrier_fraction_scan_eq19": [
        (
            r"λπ-constrained three-range \+ channel split \(k,q\) \+ triplet barrier-fraction scan",
            "λπ拘束3レンジ + チャネル分離(k,q) + 三重項障壁比率走査",
        ),
        (
            r"eq19 \(Nijmegen\) three-range \(triplet barrier-fraction scan\)",
            "eq19（Nijmegen）\n3レンジ（三重項障壁比率走査）",
        ),
        (r"Potential V\(r\) \(MeV\)", "ポテンシャル V(r)（MeV）"),
        (r"triplet \(fit B, a_t, r_t, v2t\)", "三重項（B, a_t, r_t, v2t をフィット）"),
        (r"singlet \(fit V1, V2>=0 by a_s, r_s\)", "一重項（a_s, r_sで V1, V2>=0 をフィット）"),
        (r"triplet: ERE fit \(v2 targets\)", "三重項: EREフィット（v2を目標）"),
        (r"k grid", "kグリッド"),
        (r"ERE fit", "EREフィット"),
        (r"fit/predictions − observed \(units: fm³, fm, fm³\)", "フィット/予測 − 観測（単位: fm³, fm, fm³）"),
        (r"effective-range function kcot δ \(fm−1\)", "有効レンジ関数 kcot δ（fm−1）"),
        (r"λπ-constrained three-range \+ channel split \(k,q\) \+ triplet barrier-fraction scan", "λπ拘束3レンジ + チャネル分離(k,q) + triplet障壁比率走査"),
        (r"eq19 \(Nijmegen\) three-range \(triplet barrier-fraction scan\)", "eq19 (Nijmegen) 3レンジ(triplet障壁比率走査)"),
        (r"Potential V\(r\) \(MeV\)", "ポテンシャル V(r)(MeV)"),
        (r"triplet\(B, a_t, r_t, v2t fit\)", "triplet(B, a_t, r_t, v2t フィット)"),
        (r"singlet\(a_s, r_s with V1, V2>=0 fit\)", "singlet(a_s, r_sで V1, V2>=0 フィット)"),
        (r"triplet: ERE fit \(v2 targets\)", "triplet: EREフィット(v2 targets)"),
        (r"k grid", "kグリッド"),
        (r"ERE fit", "EREフィット"),
        (r"fit/predictions − observed \(units: fm³, fm, fm³\)", "フィット/predictions − observed(単位: fm³, fm, fm³)"),
        (r"effective-range function kcot δ \(fm−1\)", "effective range関数kcot δ(fm−1)"),
    ],
    "nuclear_binding_energy_frequency_mapping_minimal_additional_physics": [
        (r"binding saturation per nucleon \(ν\)", "核子あたりの結合飽和（ν）"),
        (r"minimal additional physics under frozen thresholds", "凍結 threshold 下での最小追加物理"),
        (r"operational z-score \(abs\(z\)≤3 = pass\)", "運用 z-score（abs(z)≤3 で pass）"),
        (r"z \(σ_proxy units\)", "z（σ_proxy 単位）"),
        (r"binding saturation per nucleon \(ν\)", "核子あたり 結合飽和(ν)"),
        (r"minimal additional physics under frozen thresholds", "凍結 threshold 下atminimal additional physics"),
        (r"operational z-score \(abs\(z\)≤3 = pass\)", "operational z-score(abs(z)≤3 で pass"),
        (r"z \(σ_proxy units\)", "z(σ_proxy 単位)"),
    ],
    "nuclear_binding_energy_frequency_mapping_falsification_pack": [
        (r"falsification pack and independent cross-check", "反証条件パックと独立クロスチェック"),
        (r"A-trend consistency \(\|z_Δmedian\|<=3\)", "Aトレンド整合性（|z_Δmedian|<=3）"),
        (r"global ratio R", "グローバル比 R"),
        (r"local spacing ratio d", "局所間隔比 d"),
        (r"z_Δmedian \(log10 ratio\)", "z_Δmedian（log10比）"),
        (r"z_median \(log10 ratio\)", "z_median（log10比）"),
        (r"independent cross-check \(charge-radius kink, A_min=100\)", "独立クロスチェック（電荷半径キンク, A_min=100）"),
        (r"falsification pack and independent cross-check", "falsification pack and 独立cross-check"),
        (r"A-trend consistency \(\|z_Δmedian\|<=3\)", "Aトレンドconsistency(|z_Δmedian|<=3)"),
        (r"global ratio R", "グローバル比 R"),
        (r"local spacing ratio d", "局所間隔比 d"),
        (r"z_Δmedian \(log10 ratio\)", "z_Δmedian(log10比)"),
        (r"z_median \(log10 ratio\)", "z_median(log10比)"),
        (r"independent cross-check \(charge-radius kink, A_min=100\)", "独立クロスチェック（電荷半径キンク, A_min=100）"),
    ],
    "molecular_transitions_exomol_baseline": [
        (
            r"representative molecular-transition baseline \(two lines extracted per molecule from primary line lists\)",
            "分子遷移の代表基準（一次線リストから各分子 2 本を抽出）",
        ),
        (r"molecule", "分子"),
        (r"representative transition 1", "代表遷移 1"),
        (r"representative transition 2", "代表遷移 2"),
        (r"representative molecular-transition baseline \(two lines extracted per molecule from primary line lists\)", "分子遷移 代表基準(一次線リストから各分子 2 本 Extracted)"),
        (r"molecule", "分子"),
        (r"representative transition 1", "代表遷移 1"),
        (r"representative transition 2", "代表遷移 2"),
    ],
    "v2_trial3_weak_checkpoint_summary": [
        (r"theory baseline from coupled-localization conditions of W/Z mass anchor points", "W/Z 質量基準点の結合局在条件による理論基準"),
        (r"errors of mass anchor points on the same family", "同一系列上の質量基準点の誤差"),
        (r"coupled localization", "結合局在"),
        (r"closeout scale: about 0\.002%", "closeout 規模: 約 0.002%"),
        (r"single-component clips are reclassified", "単成分 clip は再分類"),
        (r"physical localization is judged by coupled eigenmodes", "物理局在は結合固有モードで判定"),
        (r"numerical solver and spectral scale", "数値解法とスペクトルの尺度"),
        (r"count \(log axis\)", "件数（対数軸）"),
        (r"relative error \[%\]", "相対誤差 [%]"),
        (r"charge-window extension used in the theory baseline", "理論基準で使う電荷窓拡張"),
    ],
    "condensed_silicon_thermal_expansion_gruneisen_two_branch_model": [
        (r"Silicon thermal expansion: two-branch Debye–Grüneisen model", "シリコン熱膨張: 2枝 Debye–Grüneisen モデル"),
        (r"two-branch model: α≈A1·Cv\(θ1\)\+A2·Cv\(θ2\)", "2枝モデル: α≈A1·Cv(θ1)+A2·Cv(θ2)"),
        (r"Silicon thermal expansion: two-branch Debye–Grüneisen model", "Siliconthermal expansion: 2枝 Debye–Grüneisen model"),
        (r"two-branch model: α≈A1·Cv\(θ1\)\+A2·Cv\(θ2\)", "2枝model: α≈A1·Cv(θ1)+A2·Cv(θ2)"),
        (r"branch 1 contribution", "枝1寄与"),
        (r"branch 2 contribution", "枝2寄与"),
    ],
    "condensed_silicon_thermal_expansion_gruneisen_cp_proxy_gammaT_bulkmodulus_model": [
        (r"Silicon thermal expansion: Cp-proxy Grüneisen \(tanh γ\(T\)\) ansatz check", "シリコン 熱膨張: Cp-proxy Grüneisen (tanh γ(T)) ansatz check"),
    ],
    "condensed_silicon_thermal_expansion_gruneisen_debye_einstein_two_branch_model": [
        (r"Silicon thermal expansion: Debye\+Einstein×2 \(two optical branches\) ansatz check", "シリコン 熱膨張: Debye+Einstein×2 (two optical branches) ansatz check"),
    ],
    "condensed_silicon_thermal_expansion_gruneisen_ioffe_phonon_anchors_three_einstein_model": [
        (r"Si thermal expansion: Debye\+Einstein×3 with θE anchors from Ioffe phonon frequencies", "Si 熱膨張: Debye+Einstein×3 with θE anchors from Ioffe フォノン frequencies"),
        (r"pred \(Debye\+Einstein×3\)", "予測 (Debye+Einstein×3)"),
        (r"z = \(pred−obs\)/σ_fit", "z = (予測−obs)/σ_fit"),
        (r"E1 \(anchor\)=ν_TA\(X_3\): θE1=216\.0 K", "E1(アンカー)=ν_TA(X_3): θE1=216.0 K"),
        (r"E2 \(anchor\)=ν_TO\(X_4\): θE2=667\.1 K", "E2(アンカー)=ν_TO(X_4): θE2=667.1 K"),
        (r"E3 \(anchor\)=ν_TO\(L_3'\): θE3=705\.5 K", "E3(アンカー)=ν_TO(L_3'): θE3=705.5 K"),
    ],
}

_BASE_DISPLAY_OVERRIDES_EN: list[tuple[str, str]] = [
    (r"alpha\^\{-1\}", "α⁻¹"),
    (r"alpha\^-1", "α⁻¹"),
    (r"Δ\(alpha\^-1\)", "Δ(α⁻¹)"),
    (r"10\^-8 / K", "10⁻⁸ K⁻¹"),
    (r"10\^4 Hz", "10⁴ Hz"),
    (r"H_I_", "H I "),
    (r"He_I_", "He I "),
    (r"log10\(B_pred/B_obs\)", "log10(pred/obs ratio)"),
    (r"median\(B_pred/B_obs\)", "median(pred/obs ratio)"),
    (r"B_pred/B_obs", "pred/obs ratio"),
    (r"C_required/\(A-1\)", "required coherence /(A-1)"),
    (r"A_eff", "Aeff"),
    (r"A_inf", "Ainf"),
    (r"σ_fit", "σ fit"),
    (r"α_pred", "pred"),
    (r"α_obs", "obs"),
]


# 関数: `_ordered_patterns` の入出力契約と処理意図を定義する。
def _ordered_patterns(
    patterns: Iterable[tuple[str, str]],
    *,
    ignore_case: bool,
) -> list[tuple[re.Pattern[str], str]]:
    ordered = sorted(patterns, key=lambda item: len(item[0]), reverse=True)
    flags = re.IGNORECASE if ignore_case else 0
    return [(re.compile(pattern, flags), repl) for pattern, repl in ordered]


# 関数: `_pattern_to_display` の入出力契約と処理意図を定義する。

def _pattern_to_display(pattern: str) -> str:
    text = pattern
    replacements = (
        (r"\b", ""),
        (r"\(", "("),
        (r"\)", ")"),
        (r"\[", "["),
        (r"\]", "]"),
        (r"\+", "+"),
        (r"\-", "-"),
        (r"\|", "|"),
    )
    for src, dst in replacements:
        text = text.replace(src, dst)

    text = re.sub(r"\\([.^$*+?{}[\]|()])", r"\1", text)
    text = re.sub(r"[ ]{2,}", " ", text)
    return text.strip()


# 関数: `_localized_to_regex` の入出力契約と処理意図を定義する。

def _localized_to_regex(localized: str) -> str:
    parts: list[str] = []
    for ch in localized:
        if ch in {" ", "\n", "\t"}:
            parts.append(r"\s*")
        elif ch in {"（", "("}:
            parts.append(r"[（(]")
        elif ch in {"）", ")"}:
            parts.append(r"[）)]")
        elif ch in {"：", ":"}:
            parts.append(r"[:：]\s*")
        elif ch in {"；", ";"}:
            parts.append(r"[;；]\s*")
        elif ch in {"，", ",", "、"}:
            parts.append(r"[,，、]\s*")
        elif ch in {"-", "−", "–"}:
            parts.append(r"[-−–]")
        else:
            parts.append(re.escape(ch))

    return "".join(parts)


# 関数: `_reverse_patterns` の入出力契約と処理意図を定義する。

def _reverse_patterns(patterns: Iterable[tuple[str, str]]) -> list[tuple[re.Pattern[str], str]]:
    reverse_pairs = [(_localized_to_regex(localized), _pattern_to_display(pattern)) for pattern, localized in patterns]
    return _ordered_patterns(reverse_pairs, ignore_case=False)


_BASE_PATTERNS_JA = _ordered_patterns(_BASE_GLOSSARY, ignore_case=True)
_BASE_PATTERNS_EN = _reverse_patterns(_BASE_GLOSSARY)
_BASE_DISPLAY_PATTERNS_EN = _ordered_patterns(_BASE_DISPLAY_OVERRIDES_EN, ignore_case=False)
_STEM_PATTERNS_JA = {
    stem: _ordered_patterns(patterns, ignore_case=True) for stem, patterns in _STEM_SPECIFIC_GLOSSARY.items()
}
_STEM_PATTERNS_EN = {stem: _reverse_patterns(patterns) for stem, patterns in _STEM_SPECIFIC_GLOSSARY.items()}
_FONT_CACHE: str | None = None


# 関数: `_reserve_suptitle_gap_if_needed` の入出力契約と処理意図を定義する。
def _reserve_suptitle_gap_if_needed(fig, *, profile_name: str) -> None:
    normalized = str(profile_name or "").strip().lower()
    if normalized not in {
        "part3_quantum",
        "part3a_quantum_foundations",
        "part3b_quantum_verification",
    }:
        return

    suptitle = getattr(fig, "_suptitle", None)
    if suptitle is None:
        return

    try:
        title_text = str(suptitle.get_text() or "").strip()
    except Exception:
        return

    if not title_text:
        return

    axes = [ax for ax in getattr(fig, "axes", []) if getattr(ax, "get_visible", lambda: True)()]
    if not axes:
        return

    # 関数: `_union_bbox` の入出力契約と処理意図を定義する。

    def _union_bbox(items):
        valid = [bbox for bbox in items if bbox is not None]
        if not valid:
            return None

        x0 = min(float(bbox.x0) for bbox in valid)
        y0 = min(float(bbox.y0) for bbox in valid)
        x1 = max(float(bbox.x1) for bbox in valid)
        y1 = max(float(bbox.y1) for bbox in valid)
        return (x0, y0, x1, y1)

    # 関数: `_group_axes_by_row` の入出力契約と処理意図を定義する。

    def _group_axes_by_row(visible_axes):
        ordered = sorted(visible_axes, key=lambda ax: (-float(ax.get_position().y0), float(ax.get_position().x0)))
        rows: list[list[object]] = []
        tolerance = 0.03
        for ax in ordered:
            y0 = float(ax.get_position().y0)
            if rows and abs(float(rows[-1][0].get_position().y0) - y0) <= tolerance:
                rows[-1].append(ax)
            else:
                rows.append([ax])

        return rows

    if normalized == "part3b_quantum_verification":
        min_suptitle_gap = 0.075
        min_interrow_gap = 0.055
        min_top = 0.18
        max_iter = 10
    else:
        min_suptitle_gap = 0.020
        min_interrow_gap = 0.014
        min_top = 0.44
        max_iter = 4

    try:
        for _ in range(max_iter):
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            suptitle_bbox = suptitle.get_window_extent(renderer=renderer).transformed(fig.transFigure.inverted())
            rows = _group_axes_by_row(axes)
            if not rows:
                return

            top_row_boxes = [
                ax.get_tightbbox(renderer).transformed(fig.transFigure.inverted())
                for ax in rows[0]
            ]
            top_row_union = _union_bbox(top_row_boxes)
            if top_row_union is None:
                return

            top_gap = float(suptitle_bbox.y0) - float(top_row_union[3])
            need_top_delta = 0.0
            need_hspace_delta = 0.0
            if top_gap < min_suptitle_gap:
                need_top_delta = max(need_top_delta, min_suptitle_gap - top_gap + 0.018)

            for upper_row, lower_row in zip(rows, rows[1:]):
                upper_boxes = [
                    ax.get_tightbbox(renderer).transformed(fig.transFigure.inverted())
                    for ax in upper_row
                ]
                upper_union = _union_bbox(upper_boxes)
                if upper_union is None:
                    continue

                lower_title_boxes = []
                for ax in lower_row:
                    try:
                        title_text = str(ax.get_title() or "").strip()
                    except Exception:
                        title_text = ""

                    if not title_text:
                        continue

                    try:
                        title_box = ax.title.get_window_extent(renderer=renderer).transformed(fig.transFigure.inverted())
                    except Exception:
                        continue

                    lower_title_boxes.append(title_box)

                lower_union = _union_bbox(lower_title_boxes)
                if lower_union is None:
                    continue

                inter_gap = float(upper_union[1]) - float(lower_union[3])
                if inter_gap < min_interrow_gap:
                    need_hspace_delta = max(need_hspace_delta, min_interrow_gap - inter_gap + 0.020)

            if need_top_delta <= 0.0 and need_hspace_delta <= 0.0:
                return

            subplotpars = fig.subplotpars
            avg_axes_height = sum(float(ax.get_position().height) for ax in axes) / max(len(axes), 1)
            hspace_units = 0.0
            if need_hspace_delta > 0.0 and avg_axes_height > 1e-6:
                hspace_units = need_hspace_delta / avg_axes_height

            new_top = max(min_top, float(subplotpars.top) - need_top_delta)
            new_hspace = float(subplotpars.hspace) + hspace_units
            fig.subplots_adjust(top=new_top, hspace=new_hspace)
    except Exception:
        return


# 関数: `_normalize_locale` の入出力契約と処理意図を定義する。

def _normalize_locale(raw: str | None, *, default: str = "ja") -> str:
    if raw is None or not raw.strip():
        return default

    normalized = raw.strip().lower()
    if normalized in {"ja", "jp", "japanese", "ja-jp", "ja_jp"}:
        return "ja"

    if normalized in {"en", "english", "en-us", "en_us", "en-gb", "en_gb"}:
        return "en"

    if normalized in {"source", "off", "none"}:
        return "source"

    return default


# 関数: `get_figure_language` の入出力契約と処理意図を定義する。

def get_figure_language(*, default: str = "ja") -> str:
    for env_name in _LOCALE_ENV_NAMES:
        raw = os.getenv(env_name, "").strip()
        if raw:
            return _normalize_locale(raw, default=default)

    if os.getenv("WAVEP_MPL_FORCE_JA_TEXT", "").strip().lower() in _TRUE_VALUES:
        return "ja"

    return default


# 関数: `_normalize_spacing` の入出力契約と処理意図を定義する。

def _normalize_spacing(text: str, *, target: str) -> str:
    text = re.sub(r"[ ]{2,}", " ", text)
    if target == "ja":
        text = re.sub(r"： ", "：", text)
        text = re.sub(r"\( ", "(", text)
        text = re.sub(r" \)", ")", text)
    elif target == "en":
        text = text.replace("（", "(").replace("）", ")")
        text = text.replace("：", ": ")
        text = text.replace("，", ", ").replace("、", ", ")
        text = text.replace("。", ". ")
        text = re.sub(r"\s+\.", ".", text)
        text = re.sub(r"\s+,", ",", text)

    return text.strip()


# 関数: `_target_patterns` の入出力契約と処理意図を定義する。

def _target_patterns(stem: str | None, *, target: str) -> tuple[list[tuple[re.Pattern[str], str]], list[tuple[re.Pattern[str], str]]]:
    if target == "en":
        base_patterns = _BASE_PATTERNS_EN + _BASE_DISPLAY_PATTERNS_EN
        stem_patterns = _STEM_PATTERNS_EN.get(stem or "", [])
        return stem_patterns, base_patterns

    if target == "ja":
        base_patterns = _BASE_PATTERNS_JA
        stem_patterns = _STEM_PATTERNS_JA.get(stem or "", [])
        return stem_patterns, base_patterns

    return [], []


# 関数: `_translate_segment` の入出力契約と処理意図を定義する。

def _translate_segment(text: str, *, stem: str | None, target: str) -> str:
    if target == "source":
        return text

    result = text
    stem_patterns, base_patterns = _target_patterns(stem, target=target)
    for pattern, repl in stem_patterns:
        result = pattern.sub(lambda _m, replacement=repl: replacement, result)

    for pattern, repl in base_patterns:
        result = pattern.sub(lambda _m, replacement=repl: replacement, result)

    return _normalize_spacing(result, target=target)


# 関数: `translate_plot_text` の入出力契約と処理意図を定義する。

def translate_plot_text(text: str, *, stem: str | None = None, target: str | None = None) -> str:
    if not text:
        return text

    active_target = _normalize_locale(target, default=get_figure_language())
    parts = _MATH_TOKEN_RE.split(text)
    translated: list[str] = []
    for idx, part in enumerate(parts):
        if idx % 2 == 1:
            translated.append(part)
            continue

        translated.append(_translate_segment(part, stem=stem, target=active_target))

    return "".join(translated)


# 関数: `_pick_japanese_font` の入出力契約と処理意図を定義する。

def _pick_japanese_font() -> str | None:
    global _FONT_CACHE
    if _FONT_CACHE is not None:
        return _FONT_CACHE or None

    try:
        from scripts.utils.plot_style import resolve_wavep_cjk_font_family

        preferred_name = str(os.getenv("WAVEP_MPL_CJK_FONT", "")).strip() or None
        resolved = resolve_wavep_cjk_font_family(preferred_name=preferred_name)
        if resolved:
            _FONT_CACHE = resolved
            return resolved
    except Exception:
        pass

    _FONT_CACHE = ""
    return None


# 関数: `_restore_original_font_if_needed` の入出力契約と処理意図を定義する。

def _restore_original_font_if_needed(artist: object) -> None:
    original_family = getattr(artist, "_wavep_original_fontfamily", None)
    if not original_family:
        return

    try:
        artist.set_fontfamily(original_family)
    except Exception:
        return


# 関数: `_apply_japanese_font_if_needed` の入出力契約と処理意図を定義する。

def _apply_japanese_font_if_needed(artist: object, text: str) -> None:
    if not _JAPANESE_CHAR_RE.search(text):
        _restore_original_font_if_needed(artist)
        return

    font_name = _pick_japanese_font()
    if not font_name:
        return

    try:
        artist.set_fontfamily(font_name)
    except Exception:
        return


# 関数: `_localize_figure_texts` の入出力契約と処理意図を定義する。

def _localize_figure_texts(fig: object, *, stem: str | None, target: str) -> None:
    try:
        from matplotlib.text import Text
    except Exception:
        return

    for artist in fig.findobj(lambda obj: isinstance(obj, Text)):
        try:
            current_text = artist.get_text()
        except Exception:
            continue

        if not isinstance(current_text, str) or not current_text.strip():
            continue

        original_text = getattr(artist, "_wavep_original_text", None)
        if not isinstance(original_text, str):
            original_text = current_text
            try:
                artist._wavep_original_text = current_text
            except Exception:
                pass

        original_family = getattr(artist, "_wavep_original_fontfamily", None)
        if original_family is None:
            try:
                artist._wavep_original_fontfamily = artist.get_fontfamily()
            except Exception:
                pass

        localized = translate_plot_text(original_text, stem=stem, target=target)
        if localized != current_text:
            try:
                artist.set_text(localized)
            except Exception:
                continue

        _apply_japanese_font_if_needed(artist, localized)


# 関数: `_vector_pdf_sidecar_enabled` の入出力契約と処理意図を定義する。

def _vector_pdf_sidecar_enabled() -> bool:
    raw = os.getenv("WAVEP_MPL_AUTOSAVE_VECTOR_PDF", "").strip().lower()
    return raw in _TRUE_VALUES


# 関数: `enable_figure_text_localization` の入出力契約と処理意図を定義する。

def enable_figure_text_localization(*, default_lang: str = "ja") -> None:
    try:
        from scripts.utils.plot_style import install_wavep_cjk_font_override

        target_lang = get_figure_language(default=default_lang)
        if not str(target_lang).strip().lower().startswith("en"):
            preferred_name = str(os.getenv("WAVEP_MPL_CJK_FONT", "")).strip() or None
            install_wavep_cjk_font_override(preferred_name=preferred_name)
    except Exception:
        pass

    try:
        from matplotlib.figure import Figure
    except Exception:
        return

    if getattr(Figure, "_wavep_figure_text_localizer_enabled", False):
        return

    original_savefig = Figure.savefig

    # 関数: `_savefig_with_localization` の入出力契約と処理意図を定義する。
    def _savefig_with_localization(self: Figure, fname, *args, **kwargs):  # type: ignore[override]
        stem = None
        target_path = None
        try:
            target_path = Path(str(fname))
            stem = target_path.stem
        except Exception:
            stem = None

        target_lang = get_figure_language(default=default_lang)
        _localize_figure_texts(self, stem=stem, target=target_lang)
        localized_target = None
        if target_path is not None:
            try:
                from scripts.utils.figure_locale_paths import localize_figure_output_path

                localized_target = localize_figure_output_path(target_path)
                localized_target.parent.mkdir(parents=True, exist_ok=True)
            except Exception:
                localized_target = target_path

        save_kwargs = dict(kwargs)
        profile_name = str(os.getenv("WAVEP_MPL_FONT_PROFILE", "")).strip()
        try:
            from scripts.utils.plot_style import (
                normalize_wavep_export_canvas_to_textwidth,
                should_wavep_export_normalize_to_textwidth,
            )
        except Exception:
            normalize_wavep_export_canvas_to_textwidth = None
            should_wavep_export_normalize_to_textwidth = None

        disable_canvas_normalize = str(os.getenv("WAVEP_MPL_DISABLE_CANVAS_NORMALIZE", "")).strip().lower()
        if (
            target_path is not None
            and disable_canvas_normalize not in {"1", "true", "yes", "on"}
            and should_wavep_export_normalize_to_textwidth is not None
            and should_wavep_export_normalize_to_textwidth(profile_name=profile_name)
        ):
            _reserve_suptitle_gap_if_needed(self, profile_name=profile_name)
            if str(save_kwargs.get("bbox_inches", "")).strip().lower() == "tight":
                save_kwargs.pop("bbox_inches", None)

            if normalize_wavep_export_canvas_to_textwidth is not None:
                normalize_wavep_export_canvas_to_textwidth(self, profile_name=profile_name)

        actual_target = localized_target if localized_target is not None else fname
        result = original_savefig(self, actual_target, *args, **save_kwargs)

        if localized_target is not None and localized_target.suffix.lower() == ".png" and not _vector_pdf_sidecar_enabled():
            pdf_target = localized_target.with_suffix(".pdf")
            pdf_kwargs = dict(save_kwargs)
            pdf_kwargs["format"] = "pdf"
            pdf_kwargs.pop("dpi", None)
            original_savefig(self, pdf_target, *args, **pdf_kwargs)

        return result

    Figure.savefig = _savefig_with_localization  # type: ignore[assignment]
    Figure._wavep_figure_text_localizer_enabled = True


# 関数: `enable_japanese_figure_localization` の入出力契約と処理意図を定義する。

def enable_japanese_figure_localization() -> None:
    enable_figure_text_localization(default_lang="ja")
