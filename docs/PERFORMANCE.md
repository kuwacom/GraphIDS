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
| r1a-long 46 epoch 実測(プロセス全体の wall) | 33 分 04 秒 | **6 分 35 秒(5.0 倍)** | 5 分 15 秒(6.3 倍) |

46 epoch の推定は 1 epoch の実測を 46 倍したもの。実測の詳細と 46 epoch での回帰判定は下の「46 epoch 前後比較」にある
((A) は合格、(B) は own bar 相対差 4.6e-2 で不合格)。

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
  own bar が 0.33% 動く。46 epoch では 4.6% まで広がり、NetSec の評価で攻撃 TPR も 1 種変わった(「46 epoch 前後比較」)
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

## 46 epoch 前後比較

r1a-long-v3(1,563,644 行)を標準プロトコルのまま 46 epoch(`graphids-redo/r1a-long-v3.yaml`・seed 24)で学習し、
基点 328cb1d・(A) 既定 838fd48・(B) `--full_neighbor_agg` を比べた。基点は非決定性の床を取るため 2 回走らせた。

- 計測条件: kuwa-01 GPU0 を単独使用(同じ GPU への相乗りなし)。MoE v3 の学習が GPU1 で稼働(GPU1 の sm 平均は基点 16.6%・
  (A) 17.2%・(B) 0.1%・基点 2 回目 9.0%、VRAM 1.5〜1.9GB)。CPU 12 コア制限・`OMP_NUM_THREADS=4`。起動時の load average は
  基点 5.7・(A) 11.2・(B) 5.1・基点 2 回目 31.6。基点 -> (A) -> (B) の順に連続実行(2026-10-14 23:28〜10-15 00:14 UTC、基点 2 回目は 00:22〜00:57)
- wall は `/usr/bin/time -v` で測った学習プロセス全体(`run_recorded.py` 経由。wandb は記録用の差し替えで、offline の init/finish 分約 0.5 秒を含まない)

### 学習時間

| 指標 | 基点(1 回目 / 2 回目) | (A) 既定 | (B) `--full_neighbor_agg` |
|---|---|---|---|
| プロセス全体の wall | 33:04 / 33:53 | **6:35** | 5:15 |
| 基点 1 回目に対する倍率 | 1 | **5.0 倍** | 6.3 倍 |
| 1 epoch 平均(46 epoch 全体) | 40.9 / 41.9 秒 | **8.33 秒** | 6.61 秒 |
| 1 epoch 平均(2 epoch 目以降) / 1 epoch 目 | 40.9 / 41.8 秒、42.6 / 47.9 秒 | 8.28 秒 / 10.5 秒 | 6.58 秒 / 7.7 秒 |
| 学習ループ(train) / test / 終了待ち | 1883 / 1.2 / 約 93 秒 | 383 / 0.87 / 約 2 秒 | 304 / 0.98 / 約 2 秒 |
| find_threshold / validate(1 epoch 平均) | 24.6 / 0.73 秒 | 0.030 / 0.30 秒 | 0.033 / 0.24 秒 |
| GPU0 sm 平均(学習区間) | 18.4 / 18.3% | **87.8%** | 81.7% |
| 最大 RSS | 3.35 / 2.99GB | 1.81GB | 1.42GB |
| ピーク VRAM | 1223MB | 1237MB | 854MB |

3 epoch からの推定(基点約 32 分・(A) 約 6.4 分・(B) 約 4.8 分)どおりの結果になった。

### 回帰判定(compare.py)

`run-case.sh` の評価(基点版のモデル定義・CPU・NetSec の collect_edge_errors 相当)で比べた。床は基点 1 回目と 2 回目の比較。

| 比較 | 重みの最大差 | per-row Spearman(val / test) | own bar 相対差 | ckpt 閾値の相対差 | loss 推移の最大相対差(train / val) | 判定 |
|---|---|---|---|---|---|---|
| 床(基点同士) | 5.04e-3 | 0.99999971 / 0.99999971 | 3.9e-5 | 1.5e-4 | 2.9e-5 / 1.9e-4 | - |
| (A) 既定 | 5.13e-4 | 0.99999986 / 0.99999986 | **2.5e-5** | 5.1e-5 | 1.1e-5 / 1.7e-4 | **合格(床と同程度)** |
| (B) `--full_neighbor_agg` | 1.72e-1 | 0.9862 / 0.9881 | **4.6e-2** | 1.5e-1 | 3.4e-2 / 1.4e-1 | 不合格 |

- 46 epoch では基点自身の揺れも 3 epoch(own bar 4.2e-7)より大きく、own bar で 3.9e-5 になる。(A) はその床の範囲に収まる
- (B) は 3 epoch(own bar 3.3e-3)から 46 epoch で 4.6e-2 まで差が広がった。標準プロトコルとの比較には使えない

### 攻撃種別 TPR(NetSec の評価)

r1a-long の test には攻撃が無いので、NetSec の `audit_graphids_native_eval.py`(graphids-verification/scripts、読み取りのみ)と
同じ評価を各 checkpoint に当てた。評価が import する GraphIDS は `PYTHONPATH` の先頭で基点版(worktree)/改良版(clone)に切り替え、
flowwatch は NetSec 本体の `src` を使った。近傍サンプリングの乱数を版間で揃えるため、各ローダの走査前に seed 24 を固定した(audit 本体は固定しない)。
GPU・fanout 32768・batch 1024・window 512。区間計時つきのラッパは kuwa-01 の `work/graphids-perf/final46/timed_audit.py`。

- 評価 CSV: `work/graphids-native/eval-native-dos-ddos-full.csv`(公式 NF-v3 の DoS/DDoS 日、攻撃 1,627,316 行・7 種)と
  `work/graphids-redo/eval-native/eval-modern-injection.csv`(自前 modern 攻撃 6,547 行・6 種)
- own bar は val benign 誤差の 99% 点、TPR@1%FPR は own bar 基準(audit の `tpr_at_1pct_fpr`)と test benign 99% 点基準の両方を見た

| 比較 | own bar | own bar 相対差 | per-row Spearman(val / test / DoS 日攻撃) | 攻撃種別 TPR(own bar / test 1%FPR) |
|---|---|---|---|---|
| 基点 1 回目 | 0.00110789 | - | - | DoS 日 7 種すべて 1.0 / LOIC-HTTP のみ 0.99932、modern 6 種すべて 1.0 |
| 基点 2 回目(床) | 0.00110791 | 1.9e-5 | 0.99999970 / 0.99999971 / 0.9999985 | 基点 1 回目と全種同値 |
| (A) 既定 | 0.00110795 | 4.8e-5 | 0.99999985 / 0.99999986 / 0.9999978 | **基点 1 回目と全種同値** |
| (B) `--full_neighbor_agg` | 0.00115963 | 4.5e-2 | 0.9863 / 0.9882 / 0.954 | own bar 基準で LOIC-HTTP が 1.0 -> 0.99932(他は同値) |

- modern 攻撃の誤差はほぼ同じ値に張り付く(中央値 0.01733)ため、per-row Spearman は版を問わず意味を持たない(床でも 0.50)。最大差は 2e-6 程度
- test benign の FPR(own bar 基準)は基点 0.014031・(A) 0.014025・(B) 0.013897

### 推論時間

NetSec の評価は推論を自前の `collect_edge_errors`(モデルの encoder と transformer を直接呼ぶ)で行い、改良版で速くした
`validate`/`test` の経路を通らない。モデル定義の差は SAGELayer の関数分割だけなので、NetSec の評価時間は変わらない。
同条件(GPU0 単独・相乗りなし・load average 約 2.8)で基点版と改良版を交互に 2 回ずつ測った平均:

| 区間 | 基点 | (A) 既定 |
|---|---|---|
| プロセス全体の wall | 2:04 / 2:07 | 2:09 / 2:06 |
| import | 4.1 秒 | 4.1 秒 |
| val/test グラフ読込(キャッシュ) + モデル読込 | 1.2 + 0.4 秒 | 1.2 + 0.6 秒 |
| val 推論(156,364 行) | 9.8 秒 | 10.0 秒 |
| test 推論(156,365 行) | 9.9 秒 | 9.7 秒 |
| DoS 日 CSV のグラフ構築(1,627,316 行) | 10.6 秒 | 10.5 秒 |
| DoS 日の推論 | 85.9 秒 | 87.5 秒 |
| modern CSV のグラフ構築 + 推論 | 0.05 + 0.06 秒 | 0.05 + 0.07 秒 |
| 最大 RSS / ピーク VRAM | 3.48GB / 40MB | 3.48GB / 40MB |

GraphIDS 本体の推論経路(学習中の val/test、上の学習時間の表)は、val 156K 行の validate が 0.73 -> 0.30 秒、
test が 1.19 -> 0.87 秒、閾値探索が 24.6 -> 0.03 秒になった。NetSec の評価を速くするには NetSec 側の変更が要る
(batch 1024 の LinkNeighborLoader が支配的。inference 調査の batch 拡大・window のバッチ化・全近傍集約の案)。

### 実行方法

kuwa-01 の `work/graphids-perf/final46/` に置いたスクリプトで実行した(`queue.sh` が学習 -> compare -> 評価の順に回す。
GPU 選択・flock・`/usr/bin/time -v`・`nvidia-smi dmon` は `train-inner.sh`/`eval-inner.sh`)。中核は次のとおり:

```bash
CFG=/mnt/sdb/main/NetSec/work/graphids-redo/r1a-long-v3.yaml   # 標準設定(46 epoch)をそのまま使う
taskset -c 20-31 bash $NEW/tools/regression/run-case.sh $BASE $BASE $CFG $W/data $OUT/base46
taskset -c 20-31 bash $NEW/tools/regression/run-case.sh $NEW $BASE $CFG $W/data $OUT/new46
$PYTHON $NEW/tools/regression/compare.py $OUT/base46 $OUT/new46 --json $OUT/cmp46.json
# 攻撃 TPR: GraphIDS を PYTHONPATH の先頭で切り替えて NetSec の評価を回す
C=$W/data/pyg_graph_data/r1a-long-v3
for v in base new; do
  repo=$BASE; [[ $v == new ]] && repo=$NEW
  env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$repo:/mnt/sdb/main/NetSec/src \
    $PYTHON $W/final46/timed_audit.py \
    --checkpoint $OUT/${v}46/model.ckpt --scaler $C/scaler.pkl --val-graph $C/val.pt --test-graph $C/test.pt \
    --attack-csv /mnt/sdb/main/NetSec/work/graphids-native/eval-native-dos-ddos-full.csv \
    /mnt/sdb/main/NetSec/work/graphids-redo/eval-native/eval-modern-injection.csv \
    --window-size 512 --fanout 32768 --batch-size 1024 --seed 24 --out-dir $OUT/eval-${v}46
done
```
