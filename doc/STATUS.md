# STATUS

- UTC: 2026-04-08T13:09:05Z
- Current official state: `vector_qball_form_factor_residual_origin_missing_action_updated_pack_trial2_v2_final_watch_positive_completed_single_frozen_action_single_alpha_global_consistency_primary_diagnostic_tables_secondary_conditional_reopen_only_next`
- Latest completed official blocks: `.6091-.6098`

## Latest result

- release commit 群を `origin/main` へ push し、tag `v0.2.0` も公開した。remote `main` は commit `a97628c`、remote tag `v0.2.0` は tag object `ac09bfa` を指している。
- push 前の dirty tree は `output/public` の temp/internal 残差と `scripts/` の未追跡残差に分かれていたため、それぞれ `git stash` で退避した。保持中の stash は `pre-tag leftover untracked scripts` と `pre-tag leftover public temp artifacts` の 2 本である。
- GitHub push 経路は HTTPS から SSH へ切り替え、`ssh.github.com:443` を使う構成へ変更した。repo local の `core.sshCommand` は `C:/Users/ogawa/.ssh/id_ed25519_wavep_release` を使う。
- push 後に root [README.md](/C:/develop/waveP/README.md) と [output/public/README.md](/C:/develop/waveP/output/public/README.md) の同期ずれを確認し、public 側だけに残っていた `W/Z ボソン質量` の説明と末尾の `数式閉包` 説明を root 正本へ取り込んだうえで、`python -B scripts/summary/sync_public_readme.py --direction root-to-public` を再実行して一致させた。scientific official state は unchanged である。
- release 後の微修正として Part V 図2の接続線と丸の重なり順を正し、丸を線の手前に固定した。さらに左右パネルの小タイトルを少し下げ、メインタイトルとの間隔を詰めた。結果は [part5_fig2_title_lower_page006-06.png](/C:/develop/waveP/output/private/summary/page_audit/part5_fig2_title_lower/part5_fig2_title_lower_page006-06.png) で確認した。
- 追加の微修正として Part V 図5下段の `P-model の予測` 注記から白背景を外した。結果は [part5_fig5_nobox_page012-12.png](/C:/develop/waveP/output/private/summary/page_audit/part5_fig5_nobox/part5_fig5_nobox_page012-12.png) で確認した。
- README の follow-up 修正として、誤ってスコアボード表へ混入していた `独立銀河回転曲線 / 量子計算実験 / 弱場重力量子` の 3 行を未来予測表へ移し、Part V の説明を `5項目` から `8項目` へ更新した。その後 `root -> public` 同期を再実行して README 正本をそろえた。
- README の再 follow-up 修正として、冒頭へ追加された α の歴史的背景段落は保持したまま、再発していた同じ 3 行の表混入と Part V の `5項目` への逆戻りを修正し、再度 `root -> public` 同期を実行した。

## Workflow note

- 今回の branch は release-prep の完了と公開 surface の最終同期であり、理論計算や再検証ではない。
- 公開運用の判断基準は「root README を正本とし、public README は必ず `root -> public` 同期で一致させる」である。

## Next

- Paper lane: release 後に stash 2 本を保持したまま解析を再開する場合は、必要な方だけ `git stash apply` で戻して branch を切る。
- Paper lane: README・release note・GitHub Releases 本文などの公開案内文面を必要に応じて整える。
- Scientific lane: no unconditional next official branch.
- Reopen scientifically only if a genuinely new native relativistic bound-state bridge for retained Halpha actualizes.
- Reopen scientifically only if a genuinely new deterministic native precision correction beyond the current tree-level baselines actualizes.
- Reopen scientifically only if a genuinely new P-model-native dressed observable bridge from `alpha_P_frozen` to observables actualizes.
