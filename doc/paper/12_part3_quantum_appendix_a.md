# 補遺A：Si 熱膨張 α(T) の ansatz 試験ログ（詳細）

本文（4.2.13）で言及した ansatz 試験ログを、公開版の可読性のため補遺として分離して掲載する。

まず、Debye–Grüneisen の最小モデル（α≈A_eff·Cv; `A_eff=γ/(B V_m)` を定数とみなす）を評価する。  
このモデルでは α は Cv と同じく常に正になるため、符号反転を含む一次固定 α(T) と整合できず棄却される。

| 指標 | 値 |
|---|---:|
| Debye θ_D（既知；4.2.11） | 622.1 K |
| A_eff（T≥200 K で最小二乗） | 1.44×10^-7 mol/J |
| 相対残差RMSE（T≥200 K） | 0.182 |
| 符号の不一致（全域） | 106 点 |

`output/quantum/condensed_silicon_thermal_expansion_gruneisen_minimal_model.png`  
上：α(T)（一次固定）と、Debye–Grüneisen 最小モデル（α≈A·Cv）の比較。  
下：比 `α/Cv` の温度依存（定数でないこと）と、A_fit（T≥200 K）の基準線。  
したがって、実際の α(T) を説明するには mode-dependent な γ や anharmonicity を含む拡張が必要になる。

つづいて、`A_eff(T)=γ(T)/(B V_m)` の温度依存を最小の形で入れた ansatz として、  
`A_eff(T)=A_inf·tanh((T−T0)/ΔT)` を採用し（T0 は観測 α(T) の負→正の交差で凍結）、ΔT と A_inf を最小二乗で決めて比較した。  
T≥50 K では符号不一致は解消するが、fit標準誤差スケール（NIST記載）に対して残差が大きく、この ansatz class は棄却される。

| 指標 | 値 |
|---|---:|
| T0（負→正；凍結） | 123.7 K |
| A_inf（tanh；T≥50 K でfit） | 1.56×10^-7 mol/J |
| ΔT（tanh；T≥50 K でfit） | 136.4 K |
| 符号の不一致（T≥50 K） | 0 点 |
| 相対残差RMSE（T≥50 K; proxy） | 0.237 |
| abs(resid)>3σ_fit（T≥50 K） | 514 点 |

`output/quantum/condensed_silicon_thermal_expansion_gruneisen_gammaT_model.png`  
上：α(T) と、tanh型の `A_eff(T)` を用いた Debye–Grüneisen 最小拡張の比較。  
下：`A_eff(T)` を “観測比 α/Cv” と “モデル” で比較し、符号反転は再現できても精度が不足することを示す。

さらに、mode-dependent γ の最小実装として、2枝（2モード）モデル  
`α(T)≈A1·Cv(T;θ1)+A2·Cv(T;θ2)` を固定し、符号反転と精度（σ_fit）を同時に要求した場合の成立性を評価した。  
ここで θ1 は Debye fit（4.2.11）で凍結し、θ2 を走査して A1/A2 は σ_fit による重み付き最小二乗で決めた。

| 指標 | 値 |
|---|---:|
| θ1（凍結；Debye） | 622.1 K |
| θ2（scan best） | 577.0 K |
| A1 | 1.60×10^-6 mol/J |
| A2 | −1.43×10^-6 mol/J |
| T0_pred（負→正） | 126.8 K |
| 符号不一致（T≥50 K） | 3 点 |
| max abs(z)（T≥50 K） | 23.8 |
| reduced χ²（T≥50 K） | 56.8 |
| abs(z)>3（T≥50 K） | 423 点 |

`output/quantum/condensed_silicon_thermal_expansion_gruneisen_two_branch_model.png`  
上：2枝モデル（破線は各枝の寄与）を含めた α(T) の比較。  
下：`z=(α_pred−α_obs)/σ_fit` を示し、2枝の単純重ね合わせでは σ_fit スケールで精度不足となる点を可視化する。

さらに、Debye（音響）に対して Einstein（光学）を 1枝だけ追加した最小モデル  
`α≈A_D·Cv_D(θ_D)+A_E·Cv_E(θ_E)` を固定し、成立性を評価した（θ_D は凍結し、θ_E を走査）。

| 指標 | 値 |
|---|---:|
| θ_D（凍結） | 622.1 K |
| θ_E（scan best） | 598.6 K |
| A_D | −1.94×10^-7 mol/J |
| A_E | 3.64×10^-7 mol/J |
| T0_pred（負→正） | 126.9 K |
| 符号不一致（T≥50 K） | 3 点 |
| max abs(z)（T≥50 K） | 23.1 |
| reduced χ²（T≥50 K） | 57.8 |
| abs(z)>3（T≥50 K） | 428 点 |

`output/quantum/condensed_silicon_thermal_expansion_gruneisen_debye_einstein_model.png`  
上：Debye+Einstein の混合モデル（破線は各枝の寄与）を含めた α(T) の比較。  
下：`z=(α_pred−α_obs)/σ_fit` を示し、光学枝を足しても σ_fit スケールでは精度不足となる点を可視化する。

最後に、`α≈A_eff(T)·Cv_Debye` の形を維持したまま、`A_eff(T)=Σ c_k T^k`（n=0..3）で滑らかな温度依存を入れる “診断” を行った。  
これは導出ではなく、「Debye Cv だけで形が足りるか」を見積もるための最小テストである。  
結果として、n=3 まで拡張しても σ_fit スケールでは棄却され、単一Debye Cv を前提にした近似は不十分である。

| 指標 | 値 |
|---|---:|
| 探索した次数 n | 0,1,2,3 |
| best（min reduced χ²） | n=3 |
| max abs(z)（T≥50 K） | 52.6 |
| reduced χ²（T≥50 K） | 380 |
| abs(z)>3（T≥50 K） | 500 点 |

`output/quantum/condensed_silicon_thermal_expansion_gruneisen_polyA_model.png`  
上：多項式 `A_eff(T)` を用いた `α≈A_eff(T)·Cv_Debye` の比較（best degree）。  
下：`z=(α_pred−α_obs)/σ_fit` を示し、滑らかな `A_eff(T)` だけでは σ_fit スケールの一致に到達しない点を可視化する。

さらに、`Cv` の形状不足だけが原因かを切り分けるため、`Cv_Debye` を一次の `Cp(T)`（JANAF）で置換した proxy を用いた。  
T≥100 K は JANAF の Cp(T) を線形補間し、T<100 K は Debye Cv を Cp(100 K) でスケールして接続して `Cp_proxy(T)` を構成した。  
その上で、`A_eff(T)=A_inf·tanh((T−T0)/ΔT)`（T0は観測の符号反転で凍結）を最小二乗でフィットした。

| 指標 | 値 |
|---|---:|
| Cp_proxy（接続） | T<100 K は Debye×scale, T≥100 K は JANAF補間 |
| scale（Cp(100)/Cv_Debye(100)） | 1.181 |
| T0（負→正；凍結） | 123.7 K |
| ΔT（tanh；T≥50 K でfit） | 126.3 K |
| A_inf | 1.54×10^-7 mol/J |
| 符号不一致（T≥50 K） | 0 点 |
| max abs(z)（T≥50 K） | 55.2 |
| reduced χ²（T≥50 K） | 499 |
| abs(z)>3（T≥50 K） | 512 点 |

`output/quantum/condensed_silicon_thermal_expansion_gruneisen_cp_proxy_gammaT_model.png`  
上：`Cp_proxy(T)` を用いた Grüneisen ansatz（tanh γ(T)）と α(T) の比較。  
下：`z=(α_pred−α_obs)/σ_fit` を示し、Cv形状を一次 Cp で置換しても σ_fit スケールでは精度不足で棄却される点を可視化する。

この時点では `A_eff(T)=γ(T)/(B V_m)` のうち B を暗黙に一定と扱っているため、  
B(T) の温度依存がどの程度効くかを独立入力として切り分けた。Ioffe DB にある Si の弾性定数から  
B(T) を構成すると、400→600 K での変化は約 −1.6% に留まり、支配要因にはなりにくい。

`output/quantum/condensed_silicon_bulk_modulus_baseline.png`  
Ioffe の弾性定数（400–873 K の線形近似）から B(T)=(C11+2C12)/3 を構成し、13–600 K に投影して固定する。  
400 K 未満は room-temperature 値で一定とし、B(T) が形状不一致の主因かどうかを切り分ける入口にする。

`output/quantum/condensed_silicon_thermal_expansion_gruneisen_cp_proxy_gammaT_bulkmodulus_model.png`  
`α≈γ(T)·Cp_proxy/(B(T)V_m)`（`γ(T)=γ_inf·tanh((T−T0)/ΔT)`）を最小パラメータでフィットした結果。  
B(T) を入れても σ_fit スケールの不一致は解消せず、棄却となる（支配要因は別にあることを示す）。

さらに、Einstein（光学）を 2枝に拡張した最小モデル  
`α≈A_D·Cv_D(θ_D)+A_E1·Cv_E(θ_E1)+A_E2·Cv_E(θ_E2)` を固定し（θ_D は凍結、θ_E1<θ_E2 を走査）、成立性を評価した。  
2枝化により残差は大きく改善し、`max abs(z)` は 3.23 まで低下したが、strict（3σ＋reduced χ²<2）をわずかに満たさず棄却となる。

| 指標 | 値 |
|---|---:|
| θ_D（凍結） | 622.1 K |
| θ_E1（scan best） | 503.5 K |
| θ_E2（scan best） | 821.7 K |
| A_D | −3.22×10^-7 mol/J |
| A_E1 | 3.81×10^-7 mol/J |
| A_E2 | 1.16×10^-7 mol/J |
| T0_pred（負→正） | 123.0 K |
| 符号不一致（T≥50 K） | 1 点 |
| max abs(z)（T≥50 K） | 3.23 |
| reduced χ²（T≥50 K） | 2.32 |
| abs(z)>3（T≥50 K） | 6 点 |

`output/quantum/condensed_silicon_thermal_expansion_gruneisen_debye_einstein_two_branch_model.png`  
上：Debye+Einstein×2 の混合モデル（破線は各枝の寄与）を含めた α(T) の比較。  
下：`z=(α_pred−α_obs)/σ_fit` を示し、2枝化で改善しても strict を満たさず棄却となる点を可視化する。

さらに、温度レンジ選択（fit 範囲）が「手続き系統」として統計量をどの程度動かすかを定量化するため、  
temperature-split の holdout を行った。train 範囲で推定したパラメータを固定したまま test 範囲へ外挿し、  
`max abs(z)` と reduced χ²（test は dof=n）を比較した。  
Einstein×2 は train では `max abs(z)<1` まで到達できる一方、test では大きく悪化し、範囲選択が支配的な系統になり得る。

| Split（train→test） | model | max abs(z) train | max abs(z) test | reduced χ² train | reduced χ² test |
|---|---:|---:|---:|---:|---:|
| A（50–300→300–600 K） | Einstein×1 | 14.5 | 50.1 | 29.5 | 1060 |
| A（50–300→300–600 K） | Einstein×2 | 0.94 | 15.4 | 0.188 | 63.1 |
| B（200–600→50–200 K） | Einstein×1 | 3.56 | 85.9 | 1.67 | 3030 |
| B（200–600→50–200 K） | Einstein×2 | 0.306 | 45.3 | 0.0058 | 562 |

`output/quantum/condensed_silicon_thermal_expansion_gruneisen_holdout_splits.png`  
上：split A/B で train/test の `max abs(z)` を比較（1=Einstein×1, 2=Einstein×2）。  
下：同じ split で reduced χ² を比較し、レンジ選択が系統になり得る点を可視化する。

最後に、phonon 側の一次拘束を入れる入口として、Ioffe DB の「代表 phonon 周波数」表から  
Einstein 温度 θ_E を離散アンカーとして固定し（TA を低周波側、光学モードを高周波側として候補集合を作り、  
fit range（T≥50 K）での重み付き SSE が最小になる 2点を選ぶ）、Debye+Einstein×2 を評価した。  
これは “導出” ではなく、**θ_E の連続走査（自由度の増殖）を抑えた上で、レンジ依存がどこまで残るか** の診断である。

| 指標 | 値 |
|---|---:|
| θ_E1（Ioffe anchor best; TA） | 216.0 K |
| θ_E2（Ioffe anchor best; optical） | 743.9 K |
| max abs(z)（T≥50 K） | 5.17 |
| reduced χ²（T≥50 K） | 4.16 |
| holdout A: max abs(z)（train→test） | 1.94 → 7.81 |

`output/quantum/condensed_silicon_thermal_expansion_gruneisen_ioffe_phonon_anchors_model.png`  
上：Ioffe 由来の θ_E アンカーで固定した Debye+Einstein×2 と α(T) の比較（破線は各枝寄与）。  
下：`z=(α_pred−α_obs)/σ_fit` を示し、θ_E を固定しても σ_fit スケールの棄却と holdout 崩壊が残る点を可視化する。

同様に、光学側の分割を 2枝→3枝（Einstein×3）へ拡張し、Ioffe 由来の離散候補（optical の組合せ）から  
fit range（T≥50 K）の SSE が最小になる 3点（θ_E1<θ_E2<θ_E3）を選んで評価した。  
結果として、fit range の `max abs(z)` は 4.75 まで改善したが、still strict 判定では棄却であり、  
holdout A（50–300→300–600 K）では test の `max abs(z)` が 15.9 まで悪化してレンジ依存が残る。

| 指標 | 値 |
|---|---:|
| θ_E1（Ioffe anchor best; TA） | 216.0 K |
| θ_E2（Ioffe anchor best; optical） | 667.1 K |
| θ_E3（Ioffe anchor best; optical） | 705.5 K |
| max abs(z)（T≥50 K） | 4.75 |
| reduced χ²（T≥50 K） | 4.13 |
| holdout A: max abs(z)（train→test） | 0.90 → 15.90 |

`output/quantum/condensed_silicon_thermal_expansion_gruneisen_ioffe_phonon_anchors_three_einstein_model.png`  
上：Ioffe 由来の θ_E アンカー（Einstein×3）で固定したモデルと α(T) の比較。  
下：`z=(α_pred−α_obs)/σ_fit` を示し、枝数を増やしても holdout の崩壊（手続き系統）が残る点を可視化する。

さらに、mode-weighting（Cvの重み）を “一次（またはそれに準ずる）” で凍結する試みとして、  
公開HTMLに含まれる phonon DOS（ω–D(ω)）の数値テーブルをキャッシュし、  
mode-count に基づく分割（half-integral / 2:1 split）で Grüneisen ansatz を評価した。  
ただしこの入力は一次文献の明示が無い proxy であり、INS/IXS の一次（raw / supp）を取得できたら差し替える。

| model（DOS拘束） | max abs(z)（T≥50 K） | reduced χ²（T≥50 K） | holdout A: max abs(z)（train→test） |
|---|---:|---:|---:|
| 2-group（acoustic/optical） | 26.5 | 65.0 | 22.7 → 19.8 |
| 3-group（TA/LA/optical） | 7.45 | 11.3 | 1.48 → 20.5 |
| 4-group（TA/LA/TO/LO） | 6.05 | 5.97 | 1.12 → 16.95 |

`output/quantum/condensed_silicon_thermal_expansion_gruneisen_phonon_dos_mode_gamma_three_group_model.png`  
上：phonon DOS（per atom）と、mode-count に基づく分割（TA/LA/optical）の位置。  
中：α(T)（一次固定）と、DOS拘束 Grüneisen（A係数のみfit）の比較。  
下：`z=(α_pred−α_obs)/σ_fit` を示し、low-T train から high-T test への外挿が崩れることを可視化する。  
したがって、**phonon DOS 形状の凍結だけでは不十分**であり、温度依存の phonon softening（ω(T)）や mode-dependent γ の一次拘束が必要になる。

さらに、anharmonicity の最小 proxy として、optical 側の周波数スケールを  
`s(T)=max(s_min, 1−f·(T/T_max))`（T_max は評価温度上限、f は grid scan で train SSE 最小）で弱く変化させる  
“optical softening（線形）” を追加し、holdout が改善するかを検査した。  
結果として、**softening（1自由度）では holdout 崩壊は解消せず**、mode-dependent γ の一次拘束が依然として必要である。

| model（softening） | f(T_max) | max abs(z)（T≥50 K） | holdout A: max abs(z)（train→test） |
|---|---:|---:|---:|
| 2-group + optical softening | 0.060 | 22.8 | 21.1 → 12.2 |
| 3-group + optical softening | 0.000 | 7.45 | 1.48 → 20.5 |

`output/quantum/condensed_silicon_thermal_expansion_gruneisen_phonon_dos_mode_gamma_three_group_optical_softening_linear_model.png`  
上：DOS拘束 3-group に、optical softening（線形；fをgrid scan）を追加した α(T) の比較。  
下：`z=(α_pred−α_obs)/σ_fit` を示し、softening を入れても test 側の崩壊が残ることを可視化する。

次に、softening の形（温度依存）自体を一次データ側で拘束する試みとして、Si の Raman 測定 ω(T)（4–623 K）を一次PDFから digitize し、
`g(T)∈[0,1]`（低温で0、高温で1）として固定した上で、`s(T)=1−f·g(T)`（fのみgrid scan）を評価した。  
結果として、2-group は f≈0.04 を選んでも `max abs(z)` は 24.3 と棄却であり、3-group は f=0 を選んで差が出ず、**holdout 崩壊は解消しない**。

| model（Raman shape softening） | f(T_max) | max abs(z)（T≥50 K） | holdout A: max abs(z)（train→test） |
|---|---:|---:|---:|
| 2-group + Raman shape | 0.040 | 24.3 | 22.48 → 18.13 |
| 3-group + Raman shape | 0.000 | 7.45 | 1.48 → 20.52 |

`output/quantum/condensed_silicon_thermal_expansion_gruneisen_phonon_dos_mode_gamma_three_group_optical_softening_raman_shape_model.png`  
上：DOS拘束 3-group に、Raman ω(T) 由来の softening 形状を追加した α(T) の比較。  
下：`z=(α_pred−α_obs)/σ_fit` を示し、形を固定しても test 側の崩壊（高温域）が残ることを可視化する。

さらに、INSベースの一次ソース（Kim et al., PRB 91, 014307 (2015)）が報告する mean shift  
（100→1500 K で ⟨Δε/ε⟩≈−0.07）を “全モード共通の ω(T) スケール” として線形 proxy で凍結し、  
DOS基底へ適用した（`--dos-softening kim2015_linear_proxy`）。  
結果として、global softening を入れても holdout 崩壊は解消せず、3-group では高温側の不一致が悪化した。

| model（global softening; Kim2015） | s(1500K) | max abs(z)（T≥50 K） | holdout A: max abs(z)（train→test） |
|---|---:|---:|---:|
| 2-group + Kim2015 | −0.07（固定） | 26.46 | 23.08 → 18.10 |
| 3-group + Kim2015 | −0.07（固定） | 8.29 | 1.69 → 23.59 |

`output/quantum/condensed_silicon_thermal_expansion_gruneisen_phonon_dos_mode_gamma_three_group_dos_softening_kim2015_linear_model.png`  
上：DOS拘束 3-group に、Kim2015 の mean shift から凍結した global ω(T) スケールを適用した α(T) の比較。  
下：`z=(α_pred−α_obs)/σ_fit` を示し、global scaling だけでは test 側の崩壊が残ることを可視化する。


