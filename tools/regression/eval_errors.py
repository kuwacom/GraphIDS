"""学習済み checkpoint から val/test の edge 単位再構成誤差を出す(NetSec 互換)

NetSec の flowwatch.evaluation.graphids_eval.collect_edge_errors と同じ手順
(LinkNeighborLoader shuffle=False・window ごとに transformer を 1 本ずつ forward・
特徴量平均の二乗誤差)で edge 順の誤差を作る。比較したい checkpoint 同士は同じ
--repo(モデル定義)・同じ seed・同じデバイスで評価して、評価側の揺れを揃える

使い方:
    python tools/regression/eval_errors.py --repo <checkout> --checkpoint <ckpt> \
        --config <yaml> --data_dir <dir> --out <npz> [--csv <学習CSV>] [--device cpu]

--csv を与えると学習時と同じ stratified split を再現して攻撃種別を付ける
(無ければラベルから Benign/Malicious を付ける)
"""

import argparse
import os
import sys

import numpy as np
import torch
import yaml


def _load_config(path):
    with open(path) as f:
        raw = yaml.safe_load(f)
    return {
        k: v["value"] if isinstance(v, dict) and "value" in v else v
        for k, v in raw.items()
    }


def _bare_dataset(dataset_cls, cfg, seed):
    # CSV の読み込みやキャッシュ構築をせずに、キャッシュ名と split のメソッドだけ使う
    ds = object.__new__(dataset_cls)
    ds.name = cfg["dataset"]
    ds.seed = seed
    ds.fraction = cfg.get("fraction")
    ds.split_mode = cfg.get("split_mode", "stratified")
    ds.distribution_segment = cfg.get("distribution_segment", "pre_shift")
    return ds


def _attack_names(csv_path, cfg, ds, val_labels, test_labels):
    import pandas as pd

    # NetFlowDataset._process と同じ前処理で split だけを再現する
    seed = ds.seed
    df = pd.read_csv(csv_path)
    if cfg.get("fraction") is not None:
        df = df.groupby(by="Attack").sample(frac=cfg["fraction"], random_state=seed)
    x = df.drop(columns=["Attack", "Label"])
    y = df[["Attack", "Label"]]
    x = x.replace([np.inf, -np.inf], np.nan).fillna(0)
    df = pd.concat([x, y], axis=1)
    _, df_val, df_test = ds._split_dataframe(df)
    for name, part, labels in (
        ("val", df_val, val_labels),
        ("test", df_test, test_labels),
    ):
        if not np.array_equal(part["Label"].to_numpy(), labels):
            raise ValueError(f"{name} split の再現がキャッシュのラベルと一致しない")
    # object 配列だと np.load に allow_pickle が要るので固定長文字列にする
    return (
        df_val["Attack"].to_numpy().astype(str),
        df_test["Attack"].to_numpy().astype(str),
    )


def collect_edge_errors(model, graph, *, fanout, batch_size, window_size, device):
    from torch_geometric.loader import LinkNeighborLoader

    loader = LinkNeighborLoader(
        data=graph,
        num_neighbors=[fanout],
        edge_label_index=graph.edge_index,
        edge_label=graph.edge_labels,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    edge_errors = []
    with torch.inference_mode():
        for batch in loader:
            batch.batch_edge_couples = batch.edge_label_index.t()
            batch = batch.to(device)
            n_edges = batch.edge_label_index.shape[1]
            embeddings = model.encoder(
                batch.edge_index,
                batch.edge_attr,
                batch.batch_edge_couples,
                batch.num_nodes,
            )
            window_errors = []
            for window in torch.split(embeddings, window_size):
                recon = model.transformer(
                    window, torch.ones_like(window, dtype=torch.bool)
                )
                window_errors.append(torch.mean((recon - window) ** 2, dim=1))
            edge_errors.append(torch.cat(window_errors)[:n_edges].cpu())
    return torch.cat(edge_errors).double().numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--csv", default=None)
    parser.add_argument("--seed", type=int, default=24)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    sys.path.insert(0, os.path.abspath(args.repo))
    import warnings

    warnings.filterwarnings("ignore", message="The PyTorch API of nested tensors")
    from models.graphids import GraphIDS
    from utils.dataloaders import NetFlowDataset

    cfg = _load_config(args.config)
    ds = _bare_dataset(NetFlowDataset, cfg, args.seed)
    graph_dir = os.path.join(args.data_dir, "pyg_graph_data", ds._cache_name())
    # NetFlowDataset と同じ読み方(自前で作ったキャッシュのみを読む)
    val_graph = torch.load(os.path.join(graph_dir, "val.pt"), weights_only=False)[0]
    test_graph = torch.load(os.path.join(graph_dir, "test.pt"), weights_only=False)[0]

    edim_in = val_graph.edge_attr.shape[1]
    model = GraphIDS(
        ndim_in=val_graph.x.shape[1],
        edim_in=edim_in,
        edim_out=cfg["edim_out"],
        embed_dim=cfg["ae_embedding_dim"],
        num_heads=4,
        num_layers=cfg["num_layers"],
        window_size=cfg["window_size"],
        dropout=cfg["dropout"],
        ae_dropout=cfg["ae_dropout"],
        positional_encoding=cfg["positional_encoding"],
        agg_type=cfg["agg_type"],
        mask_ratio=cfg["mask_ratio"],
    ).to(args.device)
    chk = torch.load(args.checkpoint, weights_only=True, map_location=args.device)
    model.load_state_dict(chk["model_state_dict"])
    model.eval()

    out = {"checkpoint_threshold": np.float64(chk["threshold"])}
    for name, graph in (("val", val_graph), ("test", test_graph)):
        # fanout を超えるノードのサンプリングを checkpoint 間で揃える
        torch.manual_seed(args.seed)
        out[f"{name}_errors"] = collect_edge_errors(
            model,
            graph,
            fanout=cfg["fanout"],
            batch_size=cfg["batch_size"],
            window_size=cfg["window_size"],
            device=args.device,
        )
        out[f"{name}_labels"] = graph.edge_labels.numpy()
    if args.csv:
        out["val_attacks"], out["test_attacks"] = _attack_names(
            args.csv,
            cfg,
            ds,
            out["val_labels"],
            out["test_labels"],
        )
    else:
        for name in ("val", "test"):
            out[f"{name}_attacks"] = np.where(
                out[f"{name}_labels"] == 1, "Malicious", "Benign"
            )
    np.savez(args.out, **out)


if __name__ == "__main__":
    main()
