"""2 つの実行結果(run_recorded.py + eval_errors.py の出力)を比較して判定する

各ディレクトリには run.json・test.npz・eval.npz・model.ckpt があること

判定基準(docs/PERFORMANCE.md):
    ビット一致、または per-row 誤差 Spearman >= 0.9999・own bar 相対差 <= 1e-4・
    攻撃種別 TPR@1%FPR 同値

使い方:
    python tools/regression/compare.py <dirA> <dirB> [--json <out>]
"""

import argparse
import json
import os

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

SPEARMAN_MIN = 0.9999
OWN_BAR_REL_MAX = 1e-4


def _rel(a, b):
    a, b = float(a), float(b)
    if a == b:
        return 0.0
    return abs(a - b) / max(abs(a), abs(b), 1e-300)


def _row_stats(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        return {"shape_mismatch": [list(a.shape), list(b.shape)]}
    diff = np.abs(a - b)
    return {
        "n": int(a.size),
        "bit_equal": bool(np.array_equal(a, b)),
        "frac_diff_gt_1e-9": float((diff > 1e-9).mean()),
        "max_abs_diff": float(diff.max()) if a.size else 0.0,
        "spearman": 1.0 if np.array_equal(a, b) else float(spearmanr(a, b).statistic),
    }


def _tpr_table(errors, labels, attacks, own_bar):
    benign = errors[labels == 0]
    if benign.size == 0:
        return {}, None
    # NetSec recall_at_fpr と同じく正常誤差の 99% 点を 1%FPR の閾値にする
    fpr_bar = float(torch.quantile(torch.from_numpy(benign), 0.99))
    table = {}
    for name in sorted(set(attacks.tolist())):
        if name.lower() == "benign":
            continue
        sel = errors[attacks == name]
        scores = np.concatenate([benign, sel])
        truth = np.concatenate([np.zeros(benign.size), np.ones(sel.size)])
        table[name] = {
            "n": int(sel.size),
            "tpr_at_1pct_fpr": float((sel > fpr_bar).mean()),
            "tpr_at_own_bar": float((sel > own_bar).mean()),
            # TPR が 0 や 1 に張り付くと差が見えないので順位ベースの AUROC も比べる
            "auroc": float(roc_auc_score(truth, scores)),
        }
    return table, fpr_bar


def compare(dir_a, dir_b):
    res = {"a": dir_a, "b": dir_b}

    sd_a = torch.load(os.path.join(dir_a, "model.ckpt"), weights_only=True)
    sd_b = torch.load(os.path.join(dir_b, "model.ckpt"), weights_only=True)
    max_diff = 0.0
    bit_equal = True
    for key, ta in sd_a["model_state_dict"].items():
        tb = sd_b["model_state_dict"][key]
        if not torch.equal(ta, tb):
            bit_equal = False
            max_diff = max(max_diff, float((ta.double() - tb.double()).abs().max()))
    res["state_dict"] = {"bit_equal": bit_equal, "max_abs_diff": max_diff}
    res["checkpoint"] = {
        "epoch": [sd_a["epoch"], sd_b["epoch"]],
        "threshold": [sd_a["threshold"], sd_b["threshold"]],
        "threshold_rel_diff": _rel(sd_a["threshold"], sd_b["threshold"]),
    }

    with open(os.path.join(dir_a, "run.json")) as f:
        run_a = json.load(f)
    with open(os.path.join(dir_b, "run.json")) as f:
        run_b = json.load(f)
    hist = {}
    for key in ("train_loss", "val_loss", "val_pr_auc"):
        va = [e[key] for e in run_a["epochs"]]
        vb = [e[key] for e in run_b["epochs"]]
        hist[key] = {
            "equal": va == vb,
            "max_rel_diff": max(
                (_rel(x, y) for x, y in zip(va, vb, strict=False)), default=0.0
            ),
            "a": va,
            "b": vb,
        }
    ta, tb = run_a["thresholds"], run_b["thresholds"]
    hist["epoch_thresholds"] = {
        "equal": ta == tb,
        "max_rel_diff": max(
            (_rel(x, y) for x, y in zip(ta, tb, strict=False)), default=0.0
        ),
    }
    res["history"] = hist
    res["final_log"] = {
        k: [run_a["final"].get(k), run_b["final"].get(k)]
        for k in ("final_test_f1", "final_test_pr_auc", "test_threshold")
    }

    test_a = np.load(os.path.join(dir_a, "test.npz"))
    test_b = np.load(os.path.join(dir_b, "test.npz"))
    res["main_test_errors"] = _row_stats(test_a["errors"], test_b["errors"])
    res["main_test_errors"]["labels_equal"] = bool(
        np.array_equal(test_a["labels"], test_b["labels"])
    )

    ev_a = np.load(os.path.join(dir_a, "eval.npz"))
    ev_b = np.load(os.path.join(dir_b, "eval.npz"))
    res["eval_val_errors"] = _row_stats(ev_a["val_errors"], ev_b["val_errors"])
    res["eval_test_errors"] = _row_stats(ev_a["test_errors"], ev_b["test_errors"])
    bars = []
    tprs = []
    for ev in (ev_a, ev_b):
        benign_val = ev["val_errors"][ev["val_labels"] == 0]
        bar = float(np.quantile(benign_val, 0.99))
        bars.append(bar)
        tprs.append(
            _tpr_table(ev["test_errors"], ev["test_labels"], ev["test_attacks"], bar)
        )
    res["own_bar"] = {"a": bars[0], "b": bars[1], "rel_diff": _rel(*bars)}
    res["fpr_bar"] = {"a": tprs[0][1], "b": tprs[1][1]}
    table = {}
    tpr_equal = True
    for name in tprs[0][0]:
        ra, rb = tprs[0][0][name], tprs[1][0].get(name, {})
        table[name] = {
            "n": ra["n"],
            "tpr_at_1pct_fpr": [ra["tpr_at_1pct_fpr"], rb.get("tpr_at_1pct_fpr")],
            "tpr_at_own_bar": [ra["tpr_at_own_bar"], rb.get("tpr_at_own_bar")],
            "auroc": [ra["auroc"], rb.get("auroc")],
        }
        tpr_equal &= ra["tpr_at_1pct_fpr"] == rb.get("tpr_at_1pct_fpr")
    res["attack_tpr"] = table
    res["attack_auroc_max_abs_diff"] = max(
        (abs(r["auroc"][0] - r["auroc"][1]) for r in table.values()), default=0.0
    )

    exact = (
        bit_equal
        and res["eval_val_errors"].get("bit_equal", False)
        and res["eval_test_errors"].get("bit_equal", False)
    )
    spearman_ok = all(
        res[k].get("spearman", 0.0) >= SPEARMAN_MIN
        for k in ("eval_val_errors", "eval_test_errors")
    )
    res["verdict"] = {
        "bit_exact": exact,
        "spearman_ok": spearman_ok,
        "own_bar_ok": res["own_bar"]["rel_diff"] <= OWN_BAR_REL_MAX,
        "attack_tpr_1pct_fpr_equal": tpr_equal,
    }
    res["verdict"]["pass"] = exact or (
        spearman_ok and res["verdict"]["own_bar_ok"] and tpr_equal
    )
    return res


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dir_a")
    parser.add_argument("dir_b")
    parser.add_argument("--json", default=None)
    args = parser.parse_args()
    res = compare(args.dir_a, args.dir_b)
    text = json.dumps(res, indent=1)
    if args.json:
        with open(args.json, "w") as f:
            f.write(text)
    v = res["verdict"]
    print(
        f"state_dict bit_equal={res['state_dict']['bit_equal']} "
        f"max_abs_diff={res['state_dict']['max_abs_diff']:.3e}"
    )
    for k in ("eval_val_errors", "eval_test_errors", "main_test_errors"):
        r = res[k]
        print(
            f"{k}: bit_equal={r.get('bit_equal')} spearman={r.get('spearman')} "
            f"max_abs_diff={r.get('max_abs_diff')}"
        )
    print(
        f"own_bar a={res['own_bar']['a']:.9g} b={res['own_bar']['b']:.9g} "
        f"rel_diff={res['own_bar']['rel_diff']:.3e}"
    )
    print(
        f"checkpoint threshold rel_diff={res['checkpoint']['threshold_rel_diff']:.3e}"
    )
    for key in ("train_loss", "val_loss", "val_pr_auc", "epoch_thresholds"):
        h = res["history"][key]
        print(f"{key}: equal={h['equal']} max_rel_diff={h['max_rel_diff']:.3e}")
    diff_attacks = [
        n
        for n, r in res["attack_tpr"].items()
        if r["tpr_at_1pct_fpr"][0] != r["tpr_at_1pct_fpr"][1]
    ]
    print(f"attack TPR@1%FPR differing types: {diff_attacks}")
    print(f"attack AUROC max_abs_diff={res['attack_auroc_max_abs_diff']:.3e}")
    print(f"verdict: {v}")


if __name__ == "__main__":
    main()
