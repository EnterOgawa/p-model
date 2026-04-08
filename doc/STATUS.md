# STATUS

- UTC: 2026-04-08T12:18:00Z
- Current official state: `vector_qball_form_factor_residual_origin_missing_action_updated_pack_trial2_v2_final_watch_positive_completed_single_frozen_action_single_alpha_global_consistency_primary_diagnostic_tables_secondary_conditional_reopen_only_next`
- Latest completed official blocks: `.6091-.6098`

## Latest result

- release commit 群を `origin/main` へ push し、tag `v0.2.0` も公開した。remote `main` は commit `a97628c`、remote tag `v0.2.0` は tag object `ac09bfa` を指している。
- push 前の dirty tree は `output/public` の temp/internal 残差と `scripts/` の未追跡残差に分かれていたため、それぞれ `git stash` で退避した。保持中の stash は `pre-tag leftover untracked scripts` と `pre-tag leftover public temp artifacts` の 2 本である。
- GitHub push 経路は HTTPS から SSH へ切り替え、`ssh.github.com:443` を使う構成へ変更した。repo local の `core.sshCommand` は `C:/Users/ogawa/.ssh/id_ed25519_wavep_release` を使う。
- push 後に root [README.md](/C:/develop/waveP/README.md) と [output/public/README.md](/C:/develop/waveP/output/public/README.md) の同期ずれを確認し、public 側だけに残っていた `W/Z ボソン質量` の説明と末尾の `数式閉包` 説明を root 正本へ取り込んだうえで、`python -B scripts/summary/sync_public_readme.py --direction root-to-public` を再実行して一致させた。scientific official state は unchanged である。

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
