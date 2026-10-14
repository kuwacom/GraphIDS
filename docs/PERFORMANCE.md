# GraphIDS 学習の高速化(perf/fast-training)

基点 328cb1d に対する学習・評価の高速化の記録。NetSec 標準プロトコル(NF-v3 49 列・stratified・seed 24・
fanout 32768・batch 16384・window 512・ae_batch 64・mask_ratio 0.15・benign 学習)で計測した。

- (A) 結果不変の最適化(既定で有効): 学習経路は基点とビット単位で同じ計算・同じ乱数消費になるよう実装し、
  回帰テストで基点自身の再実行の揺れ(非決定性の床)の範囲に収まることを確認した
- (B) 結果が変わりうる最適化(オプトインのフラグ、既定は無効): `--full_neighbor_agg`

## 結論

| 指標 | 基点 328cb1d | (A) 既定 | (B) `--full_neighbor_agg` |
|---|---|---|---|
| r1a-long 1 epoch(2 epoch 目以降の平均) | 41.8 秒 | **8.4 秒** | 6.3 秒 |
| bench300k 1 epoch(同上) | 8.4 秒 | **1.4 秒** | 1.3 秒 |
| プロセス終了の待ち(ローダ後片付け) | 約 90 秒 | 約 0 秒 | 約 0 秒 |
| r1a-long 46 epoch 推定(epoch 時間 x 46) | 約 32 分 | **約 6.4 分** | 約 4.8 分 |

46 epoch の推定は 1 epoch の実測を 46 倍したもの。実測の前後比較は下の「46 epoch 前後比較」に別途追記する。

## (A) 結果不変の最適化

| コミット | 内容 | 効く区間 |
|---|---|---|
| 651ac20 | `find_threshold`(validation_f1)の 500 候補を sklearn の逐次 f1 からソート+searchsorted の一括計算に置換。候補・選択規則(最初の最大)・sklearn の macro の平均対象クラスを再現し、閾値はビット一致 | 閾値探索 r1a 25 秒 -> 0.03 秒/epoch |
| b491301 | val に正例が無い epoch は AP を計算せず sklearn と同じ 0.0 を使う。正例の無い epoch の checkpoint 保存は最終 epoch(早期停止含む)だけにする(最終的に残るファイルは同一)。val に正例がある場合の best 保存は従来どおり | 1 epoch あたり 0.1 秒未満 |
| 6ffdacc + 1285cc1 | 窓が重ならず全窓が完全なときは SequentialDataset+DataLoader を作らず reshape 1 回で窓化し、padding mask を None にする。loss は `sum/numel` で従来式とビット一致。従来の内側 DataLoader が引いていた乱数を同じ回数だけ引く(1285cc1。引かないと後続 epoch の shuffle が変わる) | 窓化のカーネル約 200 本/バッチ・評価の nested tensor 変換 |
| 1936472 | encoder が使う 3 テンソルだけを non_blocking で転送。loss・評価誤差のバッチごとの `.item()`/`.cpu()` をやめ、ループ後にまとめて取り出す(合計は従来と同順の Python float 加算) | GPU 同期 |
| 69ce628 | test 後に persistent worker を明示的に止める。インタプリタ終了時の後片付けに任せるとワーカーごとに join タイムアウト 5 秒を待ち、ローダ 3 本 x 6 ワーカーで約 90 秒かかっていた | プロセス終了 90 秒 -> 0 秒 |
| a80b66e | ワーカーに渡すグラフを edge_index だけにし、edge 特徴は device に常駐させて `edge_attr[e_id]` を gather する。サンプリング結果(サブグラフ・e_id・スーパーノードの抽出)と乱数の消費は従来と同一(r1a-long の train/val でハッシュ一致を確認)。ワーカーの index_select・プロセス間転送・pin・H2D(r1a-long で約 130MB/バッチ)がなくなる | ローダ待ち・H2D。r1a 1 epoch 16 秒 -> 8.4 秒 |

補足:

- 1 epoch の約 6 割を占めていたのは閾値探索(sklearn の f1_score を 500 回)だった。
  プロセスの固定費約 100 秒の正体は wandb ではなく DataLoader ワーカーの終了待ちだった(wandb offline の init は 0.5 秒・finish はほぼ 0 秒)
- 評価(validate/test)で padding mask を None にすると nested tensor 経路を通らなくなり、評価誤差が丸め差(1e-7 級)だけ変わる。
  学習経路はビット一致(GPU 上で loss・全勾配の一致を確認)。評価誤差の差は基点自身の再実行の揺れと同程度で、回帰テストの床に収まる
- `torch.cuda` の scatter(atomic 加算)が非決定的なため、基点同士でも重みはビット一致しない(下の「非決定性の床」)

## (B) オプトインのフラグ `--full_neighbor_agg`

1 層 mean 集約は学習パラメータに依存しないので、グラフ全体の全近傍で 1 回だけ(CPU で)集約し、バッチごとには種エッジの
index だけを渡す。近傍サンプリング・サブグラフ転送・バッチごとの scatter がなくなる。

- 種エッジの選び方は LinkNeighborLoader と同じ `DataLoader(range(エッジ数))` なので、shuffle の並び・バッチ構成は従来と一致する
  (test のラベル順の一致で確認)
- 結果が変わる理由: fanout 32768 を超えるスーパーノード(r1a-long の train で 2 個。train エッジの約 5 割が宛先)の
  ランダム抽出がなくなり全近傍の平均になる。fanout=-1 で学習したのと同じ入力になる(CPU ではビット一致をテストで確認)
- yaml 設定にキーが無ければ CLI の値が使われる(既定 false)。標準プロトコルとの比較実験以外では使わないこと
- 既定パスの checkpoint 名は区別しないので、併用時は `--checkpoint` を明示する

## ベンチ結果

kuwa-01(Tesla P40・CPU 12 コア制限 `taskset -c 20-31`・`OMP_NUM_THREADS=4`・ワーカー 6)。全 run が GPU0 を単独使用
(相乗りなし。MoE v3 の学習が GPU1 で稼働、load average 3〜12)。2026-10-14 22:15〜23:14 UTC に連続して計測。

- epoch: 2 epoch 目以降の平均(1 epoch 目はウォームアップを含む)。GPU 使用率は学習区間の `nvidia-smi dmon` の sm 平均
- 最大 RSS は `/usr/bin/time -v`(最大の単一プロセス)。ピーク VRAM は `torch.cuda.max_memory_allocated`
- プロセス wall は `run_recorded.py` 起動から終了まで(wandb なし)

### r1a-long(1,563,644 行、3 epoch)

| 段階 | 1 epoch | 1 epoch 目 | GPU sm 平均 | 最大 RSS | ピーク VRAM | プロセス wall |
|---|---|---|---|---|---|---|
| 基点 328cb1d(2 回) | 43.7 / 40.0 秒 | 43.0 / 45.5 秒 | 16.6 / 16.5% | 3.15 / 3.48GB | 1223MB | 3:55 / 3:48 |
| + find_threshold(651ac20) | 15.8 秒 | 17.7 秒 | 45.5% | 3.56GB | 1223MB | 2:32 |
| + AP/checkpoint(b491301) | 15.4 秒 | 17.7 秒 | 45.7% | 3.01GB | 1223MB | 2:30 |
| + 窓化 reshape(6ffdacc) | 15.2 秒 | 17.4 秒 | 58.0% | 2.99GB | 962MB | 2:30 |
| + 同期削減(1936472) | 16.0 秒 | 17.7 秒 | 34.2% | 2.90GB | 814MB | 2:32 |
| + ワーカー停止・乱数修正(1285cc1) | 16.1 秒 | 18.6 秒 | 40.9% | 3.28GB | 814MB | 1:04 |
| **+ edge 特徴の device gather(a80b66e)= (A) 最終** | **8.4 秒** | 9.9 秒 | **85.0%** | **1.81GB** | 1237MB | **0:37** |
| (B) `--full_neighbor_agg`(1285cc1 上) | 6.3 秒 | 7.5 秒 | 74.2% | 1.41GB | 854MB | 0:32 |

### bench300k(344,554 行、5 epoch)

| 段階 | 1 epoch | 1 epoch 目 | GPU sm 平均 | 最大 RSS | ピーク VRAM | プロセス wall |
|---|---|---|---|---|---|---|
| 基点 328cb1d(2 回) | 8.38 / 8.50 秒 | 11.3 / 10.7 秒 | 14.1 / 13.5% | 1.68GB | 1113MB | 2:27 / 2:25 |
| + find_threshold(651ac20) | 2.12 秒 | 4.3 秒 | 47.6% | 1.72GB | 1113MB | 1:54 |
| + AP/checkpoint(b491301) | 2.14 秒 | 4.4 秒 | 47.2% | 1.66GB | 1113MB | 1:55 |
| + 窓化 reshape(6ffdacc) | 1.90 秒 | 4.0 秒 | 60.6% | 1.64GB | 851MB | 1:53 |
| + 同期削減(1936472) | 1.86 秒 | 3.7 秒 | 48.0% | 1.57GB | 812MB | 1:52 |
| + ワーカー停止(69ce628) | 1.89 秒 | 4.3 秒 | 45.8% | 1.70GB | 812MB | 0:23 |
| + 乱数修正(1285cc1) | 1.94 秒 | 3.7 秒 | 45.5% | 1.60GB | 812MB | 0:23 |
| **+ edge 特徴の device gather(a80b66e)= (A) 最終** | **1.40 秒** | 3.1 秒 | 63.8% | **1.23GB** | 902MB | **0:20** |
| (B) `--full_neighbor_agg`(1285cc1 上) | 1.26 秒 | 2.4 秒 | 73.4% | 1.13GB | 829MB | 0:17 |

1936472 の時点では r1a-long の学習がローダ(ワーカーの edge_attr 切り出し)律速で、同期を減らしても短くならなかった。
a80b66e でワーカーの仕事がほぼサンプリングだけになり GPU 律速(sm 85%)に移った。

## 回帰テスト

`tools/regression/` に置いた。基点版と改良版を同じハーネス・同じ seed・同じデータで学習し、学習済み checkpoint を
**同じ評価コード(基点版のモデル定義・CPU)** で評価して比べる。

- `run_recorded.py`: 対象 checkout の `main.py` を import し、wandb の run だけを記録用に差し替えて実行する。
  epoch ごとの train_loss/val_loss/val_pr_auc・閾値・区間時間・ピーク VRAM を `run.json`、`test()` の誤差を `test.npz` に保存
- `eval_errors.py`: NetSec の `flowwatch.evaluation.graphids_eval.collect_edge_errors` と同じ手順(LinkNeighborLoader shuffle=False・
  window ごとに 1 本ずつ forward)で val/test の edge 単位誤差を出す。`--csv` で学習時の stratified split を再現して攻撃種別を付ける
- `compare.py`: 重み(state_dict)・checkpoint 閾値・loss/閾値の推移・per-row 誤差(Spearman・最大差)・own bar(benign val 誤差の 99% 点)・
  攻撃種別 TPR@1%FPR(test benign 誤差の 99% 点を閾値)・TPR@own bar・AUROC を比べて判定する
- `run-case.sh`: 学習 -> 記録 -> 評価の 1 ケース実行

判定基準: ビット一致、または per-row 誤差 Spearman >= 0.9999・own bar 相対差 <= 1e-4・攻撃種別 TPR@1%FPR 同値。

### 非決定性の床(基点同士の再実行)

| データ | 重みの最大差 | per-row 誤差 Spearman(val / test) | per-row 最大差 | own bar 相対差 | 閾値推移の相対差 |
|---|---|---|---|---|---|
| bench300k 5 epoch | 7.25e-5 | 0.99999999990 / 0.99999999994 | 1.2e-7 | 0 | 4.2e-7 |
| r1a-long 3 epoch | 6.84e-5 | 0.99999999933 / 0.99999999897 | 2.1e-7 | 4.2e-7 | 4.3e-7 |

### 改良版と基点の比較

| ケース | 重みの最大差 | Spearman(val / test) | own bar 相対差 | 攻撃種別 TPR@1%FPR | 攻撃種別 AUROC の最大差 | 判定 |
|---|---|---|---|---|---|---|
| bench300k 床(基点同士) | 7.25e-5 | 0.99999999990 / 0.99999999994 | 0 | 14 種すべて同値 | 3.7e-7 | - |
| bench300k (A) 最終 a80b66e | 7.25e-5 | 0.99999999990 / 0.99999999993 | 8.4e-10 | 14 種すべて同値 | 3.7e-7 | 合格(床と同程度) |
| r1a-long (A) 最終 a80b66e | 7.20e-5 | 0.99999999947 / 0.99999999933 | 4.3e-7 | (test に攻撃なし) | - | 合格(床と同程度) |
| bench300k (B) `--full_neighbor_agg` | 3.19e-4 | 0.9999997 / 0.9999996 | 2.9e-5 | 14 種すべて同値 | 4.0e-4 | 基準内(床の約 4〜1000 倍) |
| r1a-long (B) `--full_neighbor_agg` | 3.88e-2 | 0.99997 / 0.99997 | **3.3e-3** | (test に攻撃なし) | - | own bar が基準外 |

- bench300k 5 epoch では攻撃種別 TPR@1%FPR がほぼ全種 0(Infilteration のみ 0.003)で、TPR の一致だけでは差を検出しにくい。
  このため順位ベースの攻撃種別 AUROC(test benign 対 各攻撃)も比べている
- 途中段階では 651ac20・b491301・1285cc1 も単独で合格(床と同程度)。6ffdacc〜a2267a8 は内側 DataLoader の乱数消費を省いたことで
  2 epoch 目以降の shuffle が変わり不合格になった(own bar 相対差 6.9e-4)。1285cc1 で修正済み
- (B) は bench300k(スーパーノードの切り捨てがほぼ起きない)では差が小さく、r1a-long(train エッジの約 5 割がスーパーノード宛)では
  own bar が 0.33% 動く。r1a-long の攻撃 TPR は test に攻撃が無いので未評価(46 epoch の確認で NetSec の評価を使う)
- bench300k は val に攻撃を含むので AP による best checkpoint 選択の経路、r1a-long は val が benign のみなので最終 epoch 保存の経路を通る

## 回帰テストの実行方法

kuwa-01 の例(GPU 選択・flock・CPU 制限は運用ルールに従って外側で行う):

```bash
W=/mnt/sdb/main/NetSec/work/graphids-perf
export PYTHON=/mnt/sdb/main/NetSec/graphids-verification/.venv-gpu/bin/python
BASE=$W/base                             # git worktree add --detach $W/base 328cb1d
NEW=/mnt/sdb/main/NetSec/graphids-perf   # perf/fast-training
CFG=<num_epochs を書いた yaml>            # 例: bench300k.yaml(data/README.md の設定)
CSV=$W/data/NF-CSE-CIC-IDS2018-v3-bench300k/NF-CSE-CIC-IDS2018-v3-bench300k.csv  # r1a-long では省略
OUT=<出力 dir>

export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4
# 床: 基点を 2 回
taskset -c 20-31 bash $NEW/tools/regression/run-case.sh $BASE $BASE $CFG $W/data $OUT/base-a $CSV
taskset -c 20-31 bash $NEW/tools/regression/run-case.sh $BASE $BASE $CFG $W/data $OUT/base-b $CSV
# 改良版((B) は末尾に -- --full_neighbor_agg)
taskset -c 20-31 bash $NEW/tools/regression/run-case.sh $NEW $BASE $CFG $W/data $OUT/new $CSV
$PYTHON $NEW/tools/regression/compare.py $OUT/base-a $OUT/base-b --json $OUT/cmp-floor.json
$PYTHON $NEW/tools/regression/compare.py $OUT/base-a $OUT/new --json $OUT/cmp-new.json
```

`run-case.sh` の第 2 引数(評価 checkout)は全ケースで基点版にする。各出力 dir に `model.ckpt`(約 2MB)・`run.json`・`test.npz`・
`eval.npz` ができる。

## 試して不採用・未採用

| 案 | 理由 |
|---|---|
| ae_batch_size の拡大 | 標準プロトコルでは窓 32 個 < 64 で内側ループが常に 1 回。速度も loss も不変 |
| FP16 autocast | P40(sm_61)は FP16 演算が遅く 5 倍遅い。結果も変わる |
| bf16 autocast | 1.4 倍遅く loss も変わる |
| torch.compile | Triton が sm_70 未満に非対応で動かない |
| CUDA graphs | 学習(mask の乱数 + dropout)で capture に失敗。評価のみ可だが利得 1ms 程度 |
| SDPA バックエンドの強制 | 既定で mem-efficient が選ばれており利得なし。flash は sm_80 以上 |
| find_threshold の短絡(errors.mean() を返す) | 正例なしの val でも従来の戻り値は errors.max() で、mean は非同値。ベクトル化で 0.03 秒になったので短絡自体も入れない |
| DataLoader ワーカー数の変更 | ワーカーごとの乱数 seed の割り当てが変わり、スーパーノードの抽出が変わる((B) 扱い)。a80b66e 後はローダ律速でなくなったので不要 |
| wandb の削除・遅延 | offline の init 0.5 秒・finish ほぼ 0 秒で効果がない。NetSec は wandb の終了メッセージで学習完了を判定しているので残す |
| 評価頻度の削減 | val ローダの乱数消費が変わり以降の shuffle がずれる((B))。閾値探索の高速化で不要 |

## 46 epoch 前後比較(別途追記)

r1a-long-v3 46 epoch での基点版と改良版((A) 既定)の比較を別担当が追記する。実行例(変数は「回帰テストの実行方法」と同じ):

```bash
CFG=/mnt/sdb/main/NetSec/work/graphids-redo/r1a-long-v3.yaml   # 標準設定(46 epoch)をそのまま使う
taskset -c 20-31 bash $NEW/tools/regression/run-case.sh $BASE $BASE $CFG $W/data $OUT/base46
taskset -c 20-31 bash $NEW/tools/regression/run-case.sh $NEW $BASE $CFG $W/data $OUT/new46
$PYTHON $NEW/tools/regression/compare.py $OUT/base46 $OUT/new46 --json $OUT/cmp46.json
# 攻撃 TPR は r1a-long の test に攻撃が無いので NetSec の評価を checkpoint ごとに回す(出力は自分の dir へ)
C=$W/data/pyg_graph_data/r1a-long-v3
for v in base46 new46; do
  (cd /mnt/sdb/main/NetSec/graphids-verification && env PYTHONPATH=external/graphids:src \
    .venv-gpu/bin/python scripts/audit_graphids_native_eval.py \
    --checkpoint $OUT/$v/model.ckpt --scaler $C/scaler.pkl --val-graph $C/val.pt --test-graph $C/test.pt \
    --attack-csv /mnt/sdb/main/NetSec/work/graphids-native/eval-native-dos-ddos-full.csv \
    --window-size 512 --fanout 32768 --batch-size 1024 --output $OUT/$v-dosday.json)
done
```

- 実行時間(基点 / 改良):
- 重み・per-row 誤差・own bar・攻撃 TPR(NetSec の評価):
- 判定:
