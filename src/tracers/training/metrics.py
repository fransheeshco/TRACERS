import torch

@torch.no_grad()
def compute_metrics(np_pred, ns_logits, batch, onset_threshold=0.5):

    device = ns_logits.device

    # Ground-truth targets
    onsets = batch["onsets"].to(device).bool()
    actions = batch["actions_target"].to(device).long()

    # Valid positions: exclude padding and broken long-note boundaries
    valid = (
        batch["loss_mask"].to(device).bool()
        & batch["valid_frames"].to(device).bool()
    )

    # Model predictions
    onset_pred = np_pred >= onset_threshold
    action_pred = ns_logits.argmax(dim=-1)

    # Helper for precision, recall and F1
    def calculate_prf(tp, fp, fn):
        precision = tp.float() / (tp + fp).clamp(min=1)
        recall = tp.float() / (tp + fn).clamp(min=1)

        f1 = (
            2 * precision * recall
            / (precision + recall).clamp(min=1e-8)
        )

        return precision.item(), recall.item(), f1.item()

    # ===================================
    # 1. Onset detection metrics
    # ===================================

    onset_tp = (onset_pred & onsets & valid).sum()
    onset_fp = (onset_pred & ~onsets & valid).sum()
    onset_fn = (~onset_pred & onsets & valid).sum()

    onset_precision, onset_recall, onset_f1 = calculate_prf(
        onset_tp, onset_fp, onset_fn
    )

    # ===================================
    # 2. Action event detection metrics
    # ===================================

    gt_active = (actions != 0) & valid
    pred_active = (action_pred != 0) & valid

    action_tp = (pred_active & gt_active).sum()
    action_fp = (pred_active & ~gt_active).sum()
    action_fn = (~pred_active & gt_active).sum()

    action_precision, action_recall, action_f1 = calculate_prf(
        action_tp, action_fp, action_fn
    )

    # ===================================
    # 3. Active exact action accuracy
    # ===================================

    active_count = gt_active.sum()

    if active_count > 0:
        active_acc = (
            (action_pred[gt_active] == actions[gt_active])
            .float()
            .mean()
            .item()
        )
    else:
        active_acc = float("nan")

    # ===================================
    # 4. Event density
    # ===================================

    valid_count = valid.sum()

    if valid_count > 0:
        gt_active_rate = (
            gt_active.sum() / valid_count
        ).item()

        predicted_active_rate = (
            pred_active.sum() / valid_count
        ).item()
    else:
        gt_active_rate = float("nan")
        predicted_active_rate = float("nan")

    return {
        "onset_precision": onset_precision,
        "onset_recall": onset_recall,
        "onset_f1": onset_f1,

        "action_precision": action_precision,
        "action_recall": action_recall,
        "action_f1": action_f1,

        "active_exact_accuracy": active_acc,

        "gt_active_rate": gt_active_rate,
        "predicted_active_rate": predicted_active_rate,

        "valid_frames": valid_count.item(),
        "active_frames": active_count.item(),
    }



def init_metric_counts():
    return {
        "onset_tp": 0,
        "onset_fp": 0,
        "onset_fn": 0,

        "action_tp": 0,
        "action_fp": 0,
        "action_fn": 0,

        "active_correct": 0,
        "active_count": 0,

        "gt_active_count": 0,
        "pred_active_count": 0,
        "valid_count": 0,
    }


@torch.no_grad()
def update_metric_counts(
    counts,
    np_pred,
    ns_logits,
    batch,
    onset_threshold=0.5
):
    device = ns_logits.device

    # Ground-truth targets
    onsets = batch["onsets"].to(device).bool()
    actions = batch["actions_target"].to(device).long()

    # Valid positions
    valid = (
        batch["loss_mask"].to(device).bool()
        & batch["valid_frames"].to(device).bool()
    )

    # Predictions
    onset_pred = np_pred >= onset_threshold
    action_pred = ns_logits.argmax(dim=-1)

    # =================================
    # Onset detection
    # =================================

    counts["onset_tp"] += (
        onset_pred & onsets & valid
    ).sum().item()

    counts["onset_fp"] += (
        onset_pred & ~onsets & valid
    ).sum().item()

    counts["onset_fn"] += (
        ~onset_pred & onsets & valid
    ).sum().item()

    # =================================
    # Action event detection
    # =================================

    gt_active = actions != 0
    pred_active = action_pred != 0

    counts["action_tp"] += (
        gt_active & pred_active & valid
    ).sum().item()

    counts["action_fp"] += (
        ~gt_active & pred_active & valid
    ).sum().item()

    counts["action_fn"] += (
        gt_active & ~pred_active & valid
    ).sum().item()

    # =================================
    # Active exact action accuracy
    # =================================

    active = gt_active & valid

    counts["active_correct"] += (
        (action_pred == actions) & active
    ).sum().item()

    counts["active_count"] += active.sum().item()

    # =================================
    # Event density
    # =================================

    counts["gt_active_count"] += (
        gt_active & valid
    ).sum().item()

    counts["pred_active_count"] += (
        pred_active & valid
    ).sum().item()

    counts["valid_count"] += valid.sum().item()


def finalize_metrics(counts):

    def calculate_prf(tp, fp, fn):
        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0

        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall > 0 else 0.0
        )

        return precision, recall, f1

    onset_p, onset_r, onset_f1 = calculate_prf(
        counts["onset_tp"],
        counts["onset_fp"],
        counts["onset_fn"]
    )

    action_p, action_r, action_f1 = calculate_prf(
        counts["action_tp"],
        counts["action_fp"],
        counts["action_fn"]
    )

    active_acc = (
        counts["active_correct"] / counts["active_count"]
        if counts["active_count"] > 0 else float("nan")
    )

    gt_active_rate = (
        counts["gt_active_count"] / counts["valid_count"]
        if counts["valid_count"] > 0 else float("nan")
    )

    predicted_active_rate = (
        counts["pred_active_count"] / counts["valid_count"]
        if counts["valid_count"] > 0 else float("nan")
    )

    return {
        "onset_precision": onset_p,
        "onset_recall": onset_r,
        "onset_f1": onset_f1,

        "action_precision": action_p,
        "action_recall": action_r,
        "action_f1": action_f1,

        "active_exact_accuracy": active_acc,

        "gt_active_rate": gt_active_rate,
        "predicted_active_rate": predicted_active_rate,
    }
