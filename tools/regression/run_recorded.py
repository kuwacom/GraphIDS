"""GraphIDS の main.py を wandb なしで実行し、回帰比較用の記録を残すラッパ

基点版(328cb1d)と改良版を同じハーネスで走らせるため、対象 checkout の main を
import して wandb の run だけを記録用の偽物に差し替える。学習・評価のコードは
対象 checkout のものがそのまま動く

使い方:
    python tools/regression/run_recorded.py --repo <checkout> --out <dir> -- \
        --config <yaml> --data_dir <dir> --checkpoint <ckpt> --seed 24

出力:
    <out>/run.json   ログ(epoch ごとの loss 等)・閾値・区間時間・ピーク VRAM
    <out>/test.npz   main の test() が返した誤差とラベル(ローダ順)
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import yaml


class RecordingConfig(dict):
    """wandb.Config の属性アクセスと項目代入だけを再現する"""

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc

    def __setattr__(self, key, value):
        self[key] = value


class RecordingRun:
    def __init__(self, config):
        self.config = RecordingConfig(config)
        self.name = "recorded"
        self.logs = []
        self.log_times = []
        self.log_walls = []

    def log(self, metrics):
        self.log_times.append(time.perf_counter())
        self.log_walls.append(time.time())
        self.logs.append(
            {k: v for k, v in metrics.items() if isinstance(v, int | float | str)}
        )

    def finish(self):
        pass


class _StubPlot:
    @staticmethod
    def pr_curve(**kwargs):
        return None

    @staticmethod
    def confusion_matrix(**kwargs):
        return None


class _StubWandb:
    plot = _StubPlot


def _load_config(config):
    if not isinstance(config, str):
        return dict(config)
    with open(config) as f:
        raw = yaml.safe_load(f)
    # wandb の config yaml は {key: {value: ...}} 形式
    return {
        k: v["value"] if isinstance(v, dict) and "value" in v else v
        for k, v in raw.items()
        if not k.startswith("_")
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--out", required=True)
    args, rest = parser.parse_known_args()
    if rest and rest[0] == "--":
        rest = rest[1:]
    repo = os.path.abspath(args.repo)
    os.makedirs(args.out, exist_ok=True)
    sys.path.insert(0, repo)
    sys.argv = [os.path.join(repo, "main.py"), *rest]

    t_start = time.perf_counter()
    import torch

    import main as graphids_main
    import utils.trainers as trainers

    t_import = time.perf_counter()
    rec = {"thresholds": [], "find_threshold_sec": [], "validate_sec": []}

    orig_find = trainers.find_threshold

    def find_threshold(*a, **k):
        t = time.perf_counter()
        result = orig_find(*a, **k)
        rec["find_threshold_sec"].append(time.perf_counter() - t)
        rec["thresholds"].append(float(result))
        return result

    trainers.find_threshold = find_threshold

    orig_validate = trainers.validate

    def validate(*a, **k):
        t = time.perf_counter()
        result = orig_validate(*a, **k)
        rec["validate_sec"].append(time.perf_counter() - t)
        return result

    trainers.validate = validate

    orig_dataset = graphids_main.NetFlowDataset

    def dataset(*a, **k):
        t = time.perf_counter()
        result = orig_dataset(*a, **k)
        rec["dataset_sec"] = time.perf_counter() - t
        return result

    graphids_main.NetFlowDataset = dataset

    orig_train = graphids_main.train

    def train(*a, **k):
        rec["train_start"] = time.perf_counter()
        rec["train_start_wall"] = time.time()
        result = orig_train(*a, **k)
        rec["train_sec"] = time.perf_counter() - rec["train_start"]
        return result

    graphids_main.train = train

    test_out = {}
    orig_test = graphids_main.test

    def test(*a, **k):
        t = time.perf_counter()
        result = orig_test(*a, **k)
        rec["test_sec"] = time.perf_counter() - t
        test_out["errors"] = result[2].detach().cpu().double().numpy()
        test_out["labels"] = result[3].detach().cpu().numpy()
        return result

    graphids_main.test = test
    graphids_main.wandb = _StubWandb

    cli = graphids_main.Parser().parse_args()
    run = RecordingRun(_load_config(graphids_main.build_wandb_config(cli)))
    graphids_main.apply_cli_config(run.config, cli)
    graphids_main.ensure_config_keys(run.config)
    graphids_main.main(run)
    t_end = time.perf_counter()

    train_logs = [log for log in run.logs if "train_loss" in log]
    epoch_ends = [
        t for t, log in zip(run.log_times, run.logs, strict=True) if "train_loss" in log
    ]
    epoch_sec = []
    if "train_start" in rec:
        prev = rec["train_start"]
        for t in epoch_ends:
            epoch_sec.append(t - prev)
            prev = t
    final = next((log for log in run.logs if "final_test_f1" in log), {})
    peak_mb = (
        torch.cuda.max_memory_allocated() / (1024 * 1024)
        if torch.cuda.is_available()
        else 0.0
    )
    summary = {
        "repo": repo,
        "argv": rest,
        "config": {
            k: v
            for k, v in run.config.items()
            if isinstance(v, int | float | str | None)
        },
        "epochs": train_logs,
        "epoch_sec": epoch_sec,
        # nvidia-smi dmon 等の外部記録と突き合わせるための壁時計(epoch 終了時刻)
        "train_start_wall": rec.get("train_start_wall"),
        "epoch_end_wall": [
            w
            for w, log in zip(run.log_walls, run.logs, strict=True)
            if "train_loss" in log
        ],
        "thresholds": rec["thresholds"],
        "find_threshold_sec": rec["find_threshold_sec"],
        "validate_sec": rec["validate_sec"],
        "final": final,
        "import_sec": t_import - t_start,
        "dataset_sec": rec.get("dataset_sec"),
        "train_sec": rec.get("train_sec"),
        "test_sec": rec.get("test_sec"),
        "total_sec": t_end - t_start,
        "peak_vram_mb": peak_mb,
    }
    with open(os.path.join(args.out, "run.json"), "w") as f:
        json.dump(summary, f, indent=1)
    if test_out:
        np.savez(os.path.join(args.out, "test.npz"), **test_out)


if __name__ == "__main__":
    main()
