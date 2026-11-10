import torch

def binary_focal_loss(y, pred, gamma, pos_weight):
    """
    Biary focal loss for when y=1 is the minority class.\n
    INPUT
        gamma: factor for suppressing loss for easy examples (gamma > 1)
        pos_weight: how much to suppress loss when y=0 (0 <= pos_weight <= 1)
    """
    return -(y * (1-pred).pow(gamma) * pred.log() +
                pos_weight * (1-y) * pred.pow(gamma) * (1-pred).log()).mean()

def multi_focal_loss(y, pred, gamma, pos_weight):
    """
    Multi-class focal loss for when y=0 is the majority class.\n
    INPUT
        gamma: factor for suppressing loss for easy examples (gamma > 1)
        pos_weight: how much to suppress loss when y=0 (0 <= pos_weight <= 1)
    """
    p_y = pred[torch.arange(len(pred)), y]
    weight_mask = torch.where(y == 0, pos_weight, 1)
    return -(weight_mask * (1 - p_y).pow(gamma) * p_y.log()).mean()

def get_loss(
    np_pred,
    ns_logits,
    batch,
    np_gamma=2.0,
    np_pos_weight=0.8,
    ns_gamma=3.0,
    ns_pos_weight=0.5,
    np_loss_multiplier=7.0,
):
    device = np_pred.device

    # =====================================
    # 1. Move targets and masks to device
    # =====================================

    onsets = batch["onsets"].to(
        device=device, dtype=torch.float32
    )

    actions_target = batch["actions_target"].to(
        device=device, dtype=torch.long
    )

    loss_mask = batch["loss_mask"].to(
        device=device, dtype=torch.bool
    )

    valid_frames = batch["valid_frames"].to(
        device=device, dtype=torch.bool
    )

    # =====================================
    # 2. Validate tensor shapes
    # =====================================

    assert np_pred.shape == onsets.shape, (
        f"NP shape mismatch: {np_pred.shape} vs {onsets.shape}"
    )

    assert ns_logits.shape[:2] == actions_target.shape, (
        f"NS shape mismatch: {ns_logits.shape} vs {actions_target.shape}"
    )

    assert ns_logits.shape[-1] == 256

    # =====================================
    # 3. Combine and flatten masks
    # =====================================

    mask = (loss_mask & valid_frames).reshape(-1)

    if not mask.any():
        raise ValueError("No valid positions available for loss calculation.")

    # =====================================
    # 4. Flatten predictions and targets
    # =====================================

    np_pred_flat = np_pred.reshape(-1)
    np_target_flat = onsets.reshape(-1)

    ns_logits_flat = ns_logits.reshape(-1, 256)
    ns_target_flat = actions_target.reshape(-1)

    # =====================================
    # 5. Select valid positions
    # =====================================

    np_pred_valid = np_pred_flat[mask]
    np_target_valid = np_target_flat[mask]

    ns_logits_valid = ns_logits_flat[mask]
    ns_target_valid = ns_target_flat[mask]

    # =====================================
    # 6. Convert NS logits to probabilities
    # =====================================

    ns_probs_valid = torch.softmax(
        ns_logits_valid,
        dim=-1
    )

    # =====================================
    # 7. Calculate focal losses
    # =====================================

    np_loss = binary_focal_loss(
        np_target_valid,
        np_pred_valid,
        gamma=np_gamma,
        pos_weight=np_pos_weight,
    )

    ns_loss = multi_focal_loss(
        ns_target_valid,
        ns_probs_valid,
        gamma=ns_gamma,
        pos_weight=ns_pos_weight,
    )

    # =====================================
    # 8. Combine losses
    # =====================================

    total_loss = (
        np_loss_multiplier * np_loss
        + ns_loss
    )

    return total_loss, np_loss, ns_loss
