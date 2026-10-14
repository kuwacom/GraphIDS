import time

import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.dataloaders import SequentialDataset, collate_fn


def train(
    model,
    window_size,
    step_percent,
    ae_batch_size,
    train_loader,
    val_loader,
    start_epoch,
    num_epochs,
    optimizer,
    run,
    patience,
    checkpoint,
    device="cuda",
):
    best_pr_auc = -float("inf")
    cnt_wait = 0
    criterion = nn.MSELoss(reduction="none")
    for epoch in (pbar := tqdm(range(start_epoch + 1, num_epochs + 1), desc="Epochs")):
        batch_losses = []
        model.train()
        for batch in train_loader:
            train_emb, _ = _encode_batch(model, batch, device)
            ae_train_loader = _window_batches(
                train_emb,
                window_size,
                int(window_size * step_percent),
                ae_batch_size,
                device,
            )
            accumulated_loss = torch.tensor(0.0, device=device)
            seq_count = 0
            for ae_batch, mask in ae_train_loader:
                outputs = model.transformer(ae_batch, mask)
                # Intentionally keep gradients through the target embedding. In our runs,
                # this converged faster and worked better than a stop-gradient target.
                loss = criterion(outputs, ae_batch)
                loss = _masked_mean(loss, mask)
                accumulated_loss += loss
                seq_count += 1

            # Calculate the mean loss for the batch and backpropagate through both components
            if seq_count > 0:
                loss = accumulated_loss / seq_count
                # .item() で毎バッチ GPU を待つと次バッチの読み込みと重ならないので epoch 末にまとめる
                batch_losses.append(loss.detach())
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
        total_train_loss = _sum_scalars(batch_losses)
        total_train_loss /= len(train_loader)
        val_loss, val_errors, val_labels = validate(
            model, val_loader, ae_batch_size, window_size, device
        )
        has_positive = bool(val_labels.any())
        # 正例が無い val では sklearn の AP は常に 0.0 になるので計算を省いて同じ値を使う
        val_pr_auc = (
            average_precision_score(val_labels.cpu(), val_errors.cpu())
            if has_positive
            else 0.0
        )
        # Choose the operating threshold on the labeled validation split.
        threshold = find_threshold(val_errors, val_labels, method="validation_f1")

        # Keep saving the model if it produces the same or better validation PR-AUC
        should_save = val_pr_auc >= best_pr_auc

        # Stop training if the validation PR-AUC does not improve for a number of epochs
        if val_pr_auc > best_pr_auc:
            best_pr_auc = val_pr_auc
            cnt_wait = 0
        else:
            cnt_wait += 1
        is_last_epoch = epoch == num_epochs or cnt_wait >= patience
        # 正例が無い epoch の保存は AP 同点の上書きで、後の保存か最終 epoch の保存で
        # 必ず置き換わるため、最終的に残る checkpoint を変えずに保存回数だけ減らせる
        if should_save and (has_positive or is_last_epoch):
            model.save_checkpoint(
                checkpoint,
                optimizer=optimizer,
                epoch=epoch,
                threshold=threshold,
            )
        pbar.set_postfix(
            {
                "train_loss": total_train_loss,
                "val_loss": val_loss,
                "val_pr_auc": val_pr_auc,
            }
        )
        run.log(
            {
                "train_loss": total_train_loss,
                "val_loss": val_loss,
                "val_pr_auc": val_pr_auc,
            }
        )
        if cnt_wait >= patience:
            print("Early stopping!")
            break
    chk = torch.load(checkpoint, weights_only=True)
    model.load_state_dict(chk["model_state_dict"])
    return model, chk["threshold"]


def find_threshold(errors, labels=None, method="unsupervised", multiplier=10.0):
    if method == "unsupervised":
        median = errors.median()
        mad = (
            errors - median
        ).abs().median() * 1.4826  # Factor for normal distribution
        best_threshold = median + multiplier * mad
    elif method == "validation_f1" and labels is not None:
        # 500 候補を sklearn.f1_score で逐次評価すると val 58 万件で約 85 秒/epoch かかるため、
        # 同じ候補・同じ選択規則(最初の最大・strict '>')のまま、ソート+searchsorted で一括計算する
        errors = errors.detach().cpu()
        labels = labels.detach().cpu().long()
        thresholds = torch.linspace(errors.min(), errors.max(), steps=500)
        pos = torch.sort(errors[labels == 1]).values
        neg = torch.sort(errors[labels == 0]).values
        n1, n0 = pos.numel(), neg.numel()
        tp = (n1 - torch.searchsorted(pos, thresholds, right=True)).double()
        fp = (n0 - torch.searchsorted(neg, thresholds, right=True)).double()
        fn = n1 - tp
        tn = n0 - fp

        def _f1(t, a, b):
            den = 2 * t + a + b
            return torch.where(den > 0, 2 * t / den.clamp(min=1), torch.zeros_like(den))

        # sklearn の macro は y_true/y_pred のどちらかに現れたクラスだけを平均する
        has1 = ((tp + fn) > 0) | ((tp + fp) > 0)
        has0 = ((tn + fp) > 0) | ((tn + fn) > 0)
        macro = (_f1(tp, fp, fn) * has1 + _f1(tn, fn, fp) * has0) / (
            has1.double() + has0.double()
        )
        best_threshold = errors.mean()
        best = macro.max()
        if best > 0:
            best_threshold = thresholds[int((macro == best).nonzero()[0])].item()
    else:
        raise ValueError(
            "Invalid method for threshold finding. Use 'unsupervised' or "
            "'validation_f1' with labels."
        )
    return best_threshold


def _encode_batch(model, batch, device):
    """
    ローダのバッチから種エッジの埋め込みとラベル(CPU 上のまま)を返す
    """
    # encoder が使うテンソルだけを非同期で転送する。batch.to(device) の同期転送と
    # 未使用属性(x, e_id 等)の転送を避け、GPU 計算と次バッチの読み込みを重ねる
    edge_index = batch.edge_index.to(device, non_blocking=True)
    edge_attr = batch.edge_attr.to(device, non_blocking=True)
    edge_couples = batch.edge_label_index.to(device, non_blocking=True).t()
    embeddings = model.encoder(edge_index, edge_attr, edge_couples, batch.num_nodes)
    return embeddings, batch.edge_label


def _sum_scalars(values):
    # 従来の「total += loss.item()」と同じ順序の Python float 加算にして値を変えない
    total = 0
    for value in torch.stack(values).tolist() if values else []:
        total += value
    return total


def _window_batches(embeddings, window_size, step, ae_batch_size, device):
    """
    edge 埋め込みを transformer 入力の (窓バッチ, padding mask) の列にする

    窓が重ならず全窓が完全なときは reshape 1 回で SequentialDataset + collate_fn と
    同じ値になるので、その場合は padding mask を None にする
    """
    num_edges = embeddings.shape[0]
    if step == window_size and num_edges > 0 and num_edges % window_size == 0:
        # 毎バッチの DataLoader 構築と全 True mask の生成を省く。mask を None にすると
        # 評価時に全 False の padding mask による nested tensor 変換も走らない
        windows = embeddings.reshape(num_edges // window_size, window_size, -1)
        return [(chunk, None) for chunk in windows.split(ae_batch_size)]
    return DataLoader(
        SequentialDataset(embeddings, window=window_size, step=step, device=device),
        batch_size=ae_batch_size,
        collate_fn=collate_fn,
    )


def _masked_mean(loss, mask):
    if mask is None:
        # 全 True mask の sum(loss * mask) / sum(mask) とビット一致させるため mean() は使わない
        return torch.sum(loss) / loss.numel()
    return torch.sum(loss * mask) / torch.sum(mask)


def calculate_errors(outputs, batch, mask):
    if mask is None:
        # 全行が有効なので行の選別(と valid_mask.any() による同期)を省く
        mean_errors = torch.sum((outputs - batch) ** 2, dim=-1) / outputs.shape[-1]
        mean_errors = torch.nan_to_num(mean_errors, nan=0.0, posinf=1e6, neginf=-1e6)
        return mean_errors.reshape(-1)
    squared_errors = ((outputs - batch) ** 2) * mask
    valid_mask = mask.sum(dim=-1) > 0
    valid_counts = torch.sum(mask, dim=-1)
    mean_errors = torch.zeros_like(valid_counts, dtype=torch.float32)
    if valid_mask.any():
        mean_errors[valid_mask] = torch.sum(squared_errors, dim=-1)[
            valid_mask
        ] / torch.clamp(valid_counts[valid_mask], min=1)
    mean_errors = torch.nan_to_num(mean_errors, nan=0.0, posinf=1e6, neginf=-1e6)
    return mean_errors[valid_mask]


def _collect_errors_and_labels(
    model,
    loader,
    ae_batch_size,
    window_size,
    device,
    *,
    compute_loss=False,
):
    criterion = nn.MSELoss(reduction="none")
    model.eval()
    errors = []
    labels = []
    batch_losses = []
    with torch.inference_mode():
        for batch in loader:
            embeddings, batch_labels = _encode_batch(model, batch, device)
            labels.append(batch_labels)
            ae_loader = _window_batches(
                embeddings, window_size, window_size, ae_batch_size, device
            )
            accumulated_loss = torch.tensor(0.0, device=device)
            seq_count = 0
            for ae_batch, mask in ae_loader:
                outputs = model.transformer(ae_batch, mask)
                if compute_loss:
                    loss = criterion(outputs, ae_batch)
                    loss = _masked_mean(loss, mask)
                    accumulated_loss += loss
                seq_count += 1
                batch_errors = calculate_errors(outputs, ae_batch, mask)
                # バッチごとの .cpu() は GPU 待ちになるので最後に 1 回だけ転送する
                errors.append(batch_errors)
            if compute_loss and seq_count > 0:
                batch_losses.append(accumulated_loss / seq_count)
    total_loss = 0
    if compute_loss:
        total_loss = _sum_scalars(batch_losses)
        total_loss /= len(loader)
    labels = torch.cat(labels).cpu()
    errors = torch.cat(errors).cpu()
    return total_loss, errors, labels


def validate(model, val_loader, ae_batch_size, window_size, device):
    total_val_loss, errors, labels = _collect_errors_and_labels(
        model,
        val_loader,
        ae_batch_size,
        window_size,
        device,
        compute_loss=True,
    )
    return total_val_loss, errors, labels


def test(model, test_loader, ae_batch_size, window_size, device, threshold):
    if device == "cuda":
        torch.cuda.synchronize()
    start_time = time.perf_counter()
    _, errors, labels = _collect_errors_and_labels(
        model,
        test_loader,
        ae_batch_size,
        window_size,
        device,
    )
    if threshold is not None:
        test_pred = (errors > threshold).int()
    else:
        print("No threshold provided, using mean of errors for prediction.")
        test_pred = (errors > errors.mean()).int()
    if device == "cuda":
        torch.cuda.synchronize()
    prediction_time = time.perf_counter() - start_time
    f1 = f1_score(labels, test_pred, average="macro", zero_division=0)
    pr_auc = average_precision_score(labels, errors)
    return f1, pr_auc, errors, labels, prediction_time
