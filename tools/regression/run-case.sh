#!/usr/bin/env bash
# 回帰テストの 1 ケース(学習 -> 記録 -> NetSec 互換の誤差評価)を実行する
#
# 使い方:
#   run-case.sh <学習 checkout> <評価 checkout> <config yaml> <data_dir> <出力dir> [csv] [-- main.py 追加引数...]
#
# - 評価 checkout は比較する全ケースで同じもの(基点版)を使い、評価側の差を消す
# - csv を与えると test の攻撃種別を split 再現で付ける(r1a-long のように攻撃が無ければ省略)
# - GPU 選択・CPU 制限・flock は呼び出し側で行う
# - 環境変数 PYTHON で python を指定する(既定: python)
# - 環境変数 TRAIN_PREFIX を学習コマンドの前に付ける(例: "/usr/bin/time -v -o rss.txt")
set -euo pipefail

train_repo=$1
eval_repo=$2
config=$3
data_dir=$4
out_dir=$5
shift 5
csv_path=""
if [[ $# -gt 0 && $1 != "--" ]]; then
  csv_path=$1
  shift
fi
if [[ $# -gt 0 && $1 == "--" ]]; then
  shift
fi
python_bin=${PYTHON:-python}
tool_dir=$(cd "$(dirname "$0")" && pwd)

mkdir -p "$out_dir"
# checkpoint は CWD 相対の既定パスに書かれないよう出力先を明示する
rm -f "$out_dir/model.ckpt"
(
  cd "$out_dir"
  # shellcheck disable=SC2086
  ${TRAIN_PREFIX:-} "$python_bin" -u "$tool_dir/run_recorded.py" --repo "$train_repo" --out "$out_dir" -- \
    --config "$config" --data_dir "$data_dir" --checkpoint "$out_dir/model.ckpt" \
    --seed 24 "$@"
) > "$out_dir/train.log" 2>&1

eval_args=(--repo "$eval_repo" --checkpoint "$out_dir/model.ckpt" --config "$config"
  --data_dir "$data_dir" --out "$out_dir/eval.npz" --device cpu)
if [[ -n $csv_path ]]; then
  eval_args+=(--csv "$csv_path")
fi
"$python_bin" -u "$tool_dir/eval_errors.py" "${eval_args[@]}" > "$out_dir/eval.log" 2>&1
