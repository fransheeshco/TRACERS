import torch
import warnings

def decode_action(token, num_lanes=4):
    return [
        (int(token) // (4 ** lane)) % 4
        for lane in range(num_lanes)
    ]

def predictions_to_json(
    action_predictions,
    bpm,
    start_frame=0,
    offset_ms=0,
    lane_map=(1, 2, 3, 4),
    valid_frames=None,
):
    events = [
        {
            "type": "BPM",
            "bpm": bpm,
            "beat": 0,
        }
    ]

    # Convert predictions to Python lists
    if torch.is_tensor(action_predictions):
        action_predictions = action_predictions.detach().cpu().tolist()

    if valid_frames is not None and torch.is_tensor(valid_frames):
        valid_frames = valid_frames.detach().cpu().tolist()

    # Store unfinished long notes by model lane
    pending_slides = {}

    for t, token in enumerate(action_predictions):

        # Skip padding
        if valid_frames is not None and not valid_frames[t]:
            continue

        # Dataset targets correspond to start_frame + t + 1
        global_frame = start_frame + t + 1

        # Frame -> milliseconds -> musical beat
        time_ms = global_frame * 10
        beat = (time_ms - offset_ms) * bpm / 60000

        lanes = decode_action(token)

        for lane_idx, state in enumerate(lanes):

            lane = lane_map[lane_idx]

            # ============================
            # TAP
            # ============================

            if state == 1:
                events.append({
                    "type": "Single",
                    "lane": lane,
                    "beat": round(beat, 4),
                })

            # ============================
            # LONG NOTE START
            # ============================

            elif state == 2:
                if lane_idx in pending_slides:
                    warnings.warn(
                        f"Unclosed long note in lane {lane} "
                        f"before frame {global_frame}"
                    )

                pending_slides[lane_idx] = beat

            # ============================
            # LONG NOTE END
            # ============================

            elif state == 3:
                if lane_idx not in pending_slides:
                    warnings.warn(
                        f"Long note end without a start "
                        f"in lane {lane} at frame {global_frame}"
                    )
                    continue

                start_beat = pending_slides.pop(lane_idx)

                events.append({
                    "type": "Slide",
                    "connections": [
                        {
                            "beat": round(start_beat, 4),
                            "lane": lane,
                        },
                        {
                            "beat": round(beat, 4),
                            "lane": lane,
                        },
                    ],
                })

    # Warn about long notes without an ending
    for lane_idx, start_beat in pending_slides.items():
        warnings.warn(
            f"Long note in lane {lane_map[lane_idx]} "
            f"starting at beat {start_beat:.4f} has no ending"
        )

    # Sort events chronologically, keeping BPM first
    events[1:] = sorted(
        events[1:],
        key=lambda event: (
            event["beat"]
            if event["type"] == "Single"
            else event["connections"][0]["beat"]
        ),
    )

    return events


def export_generation_debug(
    segments,
    output_path="generation_debug.csv",
    top_k=5,
):
    """
    Export per-frame autoregressive generation diagnostics.

    Each segment must contain:
        start
        action_inputs
        action_predictions
        np_pred
        ns_logits
        actual
        valid_frames
    """

    import csv
    import torch
    from pathlib import Path

    if not segments:
        raise ValueError("No generation segments provided.")

    if top_k < 1:
        raise ValueError("top_k must be at least 1.")

    top_k = min(top_k, 256)

    # CSV columns
    fields = [
        "segment_start",
        "input_frame",
        "target_frame",
        "input_action",
        "np_prob",
        "predicted_action",
        "predicted_lanes",
        "ground_truth_action",
        "ground_truth_lanes",
        "p_empty",
        "p_active",
        "p_gt_action",
        "p_gt_action_given_active",
    ]

    for k in range(1, top_k + 1):
        fields.extend([
            f"top{k}_token",
            f"top{k}_lanes",
            f"top{k}_logit",
            f"top{k}_prob",
        ])

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    row_count = 0

    with output_path.open("w", newline="", encoding="utf-8") as file:

        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()

        for segment in segments:

            start = int(segment["start"])

            inputs = torch.as_tensor(
                segment["action_inputs"]
            ).detach().cpu().long().reshape(-1)

            predictions = torch.as_tensor(
                segment["action_predictions"]
            ).detach().cpu().long().reshape(-1)

            actual = torch.as_tensor(
                segment["actual"]
            ).detach().cpu().long().reshape(-1)

            np_probs = torch.as_tensor(
                segment["np_pred"]
            ).detach().cpu().float().reshape(-1)

            logits = torch.as_tensor(
                segment["ns_logits"]
            ).detach().cpu().float()

            valid = torch.as_tensor(
                segment["valid_frames"]
            ).detach().cpu().bool().reshape(-1)

            n = min(
                len(inputs),
                len(predictions),
                len(actual),
                len(np_probs),
                len(logits),
                len(valid),
            )

            # Convert raw logits into probabilities.
            probs = torch.softmax(logits[:n], dim=-1)

            # Top-k candidates for each frame.
            top_probs, top_tokens = torch.topk(
                probs,
                k=top_k,
                dim=-1,
            )

            top_logits = logits[:n].gather(-1, top_tokens)

            p_empty = probs[:, 0]
            p_active = 1.0 - p_empty

            # Probability assigned to the actual action.
            p_gt = probs.gather(
                -1,
                actual[:n].unsqueeze(-1),
            ).squeeze(-1)

            for t in range(n):

                if not valid[t]:
                    continue

                gt_action = int(actual[t])
                pred_action = int(predictions[t])

                row = {
                    "segment_start": start,

                    # actions_input predicts actions_target.
                    "input_frame": start + t,
                    "target_frame": start + t + 1,

                    "input_action": int(inputs[t]),
                    "np_prob": float(np_probs[t]),

                    "predicted_action": pred_action,
                    "predicted_lanes": decode_action(
                        pred_action
                    ),

                    "ground_truth_action": gt_action,
                    "ground_truth_lanes": decode_action(
                        gt_action
                    ),

                    "p_empty": float(p_empty[t]),
                    "p_active": float(p_active[t]),
                    "p_gt_action": float(p_gt[t]),

                    "p_gt_action_given_active": (
                        float(p_gt[t] / p_active[t])
                        if gt_action != 0 and p_active[t] > 0
                        else ""
                    ),
                }

                # Add top-k NS candidates.
                for k in range(top_k):

                    token = int(top_tokens[t, k])

                    row[f"top{k+1}_token"] = token
                    row[f"top{k+1}_lanes"] = (
                        decode_action(token)
                    )
                    row[f"top{k+1}_logit"] = float(
                        top_logits[t, k]
                    )
                    row[f"top{k+1}_prob"] = float(
                        top_probs[t, k]
                    )

                writer.writerow(row)
                row_count += 1

    print(f"Exported {row_count} frame records to: {output_path}")

    return row_count


@torch.inference_mode()
def generate_custom_song(
    trainer,
    audio_path,
    difficulty,
    segment_frames,
    seed_action=0,
    temperature=None,
    top_k=None,
    top_p=None,
    seed=0,
    collect_debug=False,
    context_frames=500,
):
    data = torch.load(audio_path, map_location="cpu", weights_only=True)

    # Match training alignment: predict frames 1 through T-1.
    specs = data["specs"].float()[:, 1:, :].contiguous()
    beat_phase = data["beat_phase"].long()[1:]
    beat_num = data["beat_num"].long()[1:]

    total_frames = specs.shape[1]

    assert beat_phase.shape[0] == total_frames
    assert beat_num.shape[0] == total_frames
    assert total_frames > 0
    assert 0 <= beat_phase.min() and beat_phase.max() <= 48
    assert 0 <= beat_num.min() and beat_num.max() <= 3

    if segment_frames <= 0:
        raise ValueError("segment_frames must be positive.")

    if not 0 <= context_frames < segment_frames:
        raise ValueError("context_frames must be smaller than segment_frames.")

    stride = segment_frames - context_frames
    difficulty = float(difficulty) * 0.2

    print(f"Prediction frames: {total_frames} ({total_frames * 0.01:.1f} s)")
    print(
        f"Window size: {segment_frames} | "
        f"Context: {context_frames} | Stride: {stride}"
    )

    # =====================================
    # PRECOMPUTE GLOBAL CONTEXT
    # =====================================

    model = trainer.model
    model.eval()

    device = next(model.parameters()).device

    full_global_context = None

    if hasattr(model, "prepare_global_context"):
        full_global_context = model.prepare_global_context(
            global_specs=[specs],
            global_indices=torch.tensor([0]),
            start_frames=torch.tensor([0]),
            seq_len=total_frames,
            device=device,
            dtype=specs.dtype,
        )

    # =====================================
    # GENERATION
    # =====================================

    generated = torch.empty(total_frames, dtype=torch.long)
    debug_segments = []

    written_until = 0
    window_idx = 0

    while written_until < total_frames:

        # Start the next window with previously generated context.
        start = max(0, written_until - context_frames)
        end = min(start + segment_frames, total_frames)
        length = end - start

        context_len = written_until - start

        batch = {
            "specs": specs[:, start:end].unsqueeze(0),
            "beat_phase": beat_phase[start:end].unsqueeze(0),
            "beat_num": beat_num[start:end].unsqueeze(0),
            "difficulty": torch.full(
                (1, length, 1),
                difficulty,
                dtype=torch.float32,
            ),
            "valid_frames": torch.ones(
                (1, length),
                dtype=torch.bool,
            ),
        }

        # =====================================
        # PREVIOUS ACTION CONTEXT
        # =====================================

        if start == 0:
            prefix_actions = None
        else:
            # Decoder inputs for the already-generated overlap.
            # Includes one preceding action plus context_len actions.
            prefix_actions = generated[
                start - 1 : written_until
            ].unsqueeze(0)

            if prefix_actions.size(1) != context_len + 1:
                raise RuntimeError("Previous action context is incomplete.")

        # =====================================
        # GLOBAL CONTEXT
        # =====================================

        generate_kwargs = {}

        if full_global_context is not None:
            generate_kwargs["global_context"] = (
                full_global_context[:, start:end, :]
            )

        # =====================================
        # GENERATE
        # =====================================

        results = trainer.generate(
            batch,
            seed_action=seed_action,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            seed=seed if window_idx == 0 else None,
            prefix_actions=prefix_actions,
            **generate_kwargs,
        )

        predictions = results["action_predictions"][0]

        if predictions.numel() != length:
            raise RuntimeError(
                f"Expected {length} predictions, got {predictions.numel()}."
            )

        # Discard predictions for the previously generated context.
        new_predictions = predictions[context_len:]

        expected_new = end - written_until

        if new_predictions.numel() != expected_new:
            raise RuntimeError(
                f"Expected {expected_new} new predictions, "
                f"got {new_predictions.numel()}."
            )

        generated[written_until:end] = new_predictions

        # =====================================
        # DEBUG
        # =====================================

        if collect_debug:
            debug_segments.append({
                "start": written_until + 1,
                "window_start": start + 1,
                "end": end + 1,
                "action_inputs": results["action_inputs"][0][context_len:length],
                "action_predictions": new_predictions,
                "np_pred": results["np_pred"][0][context_len:length],
                "ns_logits": results["ns_logits"][0][context_len:length],
                "valid_frames": results["valid_frames"][0][context_len:length],
            })

        print(
            f"Window {start + 1:6d}-{end:6d} | "
            f"Context: {context_len:4d} | "
            f"New: {len(new_predictions):4d} | "
            f"Active: {(new_predictions != 0).sum().item():4d}"
        )

        written_until = end
        window_idx += 1

    print(
        f"\nDone. Generated {len(generated)} frames. "
        f"Active frames: {(generated != 0).sum().item()} / {len(generated)}"
    )

    return generated, debug_segments



@torch.inference_mode()
def generate_validation_chart(
    trainer,
    loader,
    chart_id=None,
    context_frames=500,
    seed_action=None,
    temperature=None,
    top_k=None,
    top_p=None,
    seed=0,
    collect_debug=False,
):
    model = trainer.model
    model.eval()

    INPUT_KEYS = [
        "specs",
        "beat_phase",
        "beat_num",
        "difficulty",
        "valid_frames",
    ]

    # =====================================
    # COLLECT CHART WINDOWS
    # =====================================

    segments = []
    full_song_specs = None
    initial_seed_action = None

    selected_chart = chart_id
    found = False
    finished = False

    for batch in loader:
        for i, cid in enumerate(batch["chart_id"]):

            if selected_chart is None:
                selected_chart = cid

            if cid != selected_chart:
                if found:
                    finished = True
                    break
                continue

            found = True

            sample = {
                k: batch[k][i:i+1]
                for k in INPUT_KEYS
            }

            valid_length = int(
                sample["valid_frames"][0].sum().item()
            )

            segments.append({
                "start": int(batch["start_frame"][i]),
                "sample": sample,
                "actual": batch["actions_target"][i].cpu(),
                "valid_length": valid_length,
            })

            # Initial decoder action from the ground-truth chart.
            if initial_seed_action is None:
                initial_seed_action = int(
                    batch["actions_input"][i, 0].item()
                )

            # Retrieve full-song audio if provided by the collator.
            if "global_specs" in batch:
                global_idx = int(
                    batch["global_indices"][i].item()
                )

                full_song_specs = batch["global_specs"][global_idx]

        if finished:
            break

    if not segments:
        raise ValueError(f"Chart not found: {selected_chart}")

    segments.sort(key=lambda x: x["start"])

    if segments[0]["start"] != 0:
        raise ValueError("Validation windows must start at frame 0.")

    segment_length = segments[0]["sample"]["specs"].size(2)

    if not 0 <= context_frames < segment_length:
        raise ValueError("context_frames must be smaller than segment length.")

    if context_frames != 0 and not hasattr(model, "forward"):
        raise ValueError("Invalid model.")

    total_frames = max(
        seg["start"] + seg["valid_length"]
        for seg in segments
    )

    print(f"Chart: {selected_chart}")
    print(f"Windows: {len(segments)}")
    print(f"Target frames: {total_frames}")
    print(
        f"Window size: {segment_length} | "
        f"Context: {context_frames} | "
        f"Stride: {segment_length - context_frames}"
    )

    # Use ground-truth frame 0 as the initial decoder input by default.
    if seed_action is None:
        seed_action = initial_seed_action

    # =====================================
    # PRECOMPUTE GLOBAL CONTEXT
    # =====================================

    full_global_context = None

    if hasattr(model, "prepare_global_context"):

        if full_song_specs is None:
            raise ValueError(
                "Global model requires GlobalAudioCollator "
                "in the validation DataLoader."
            )

        device = next(model.parameters()).device

        full_global_context = model.prepare_global_context(
            global_specs=[full_song_specs],
            global_indices=torch.tensor([0]),
            start_frames=torch.tensor([0]),
            seq_len=full_song_specs.size(1),
            device=device,
            dtype=full_song_specs.dtype,
        )

    # =====================================
    # GENERATION
    # =====================================

    generated = torch.empty(total_frames, dtype=torch.long)
    actual = torch.empty(total_frames, dtype=torch.long)

    debug_segments = []

    written_until = 0
    window_idx = 0

    for seg in segments:

        # Stop once the entire chart has been generated.
        if written_until >= total_frames:
            break

        start = seg["start"]
        valid_length = seg["valid_length"]
        seq_len = seg["sample"]["specs"].size(2)

        end = start + valid_length

        # Ensure that windows overlap correctly and don't leave gaps.
        if start > written_until:
            raise RuntimeError(
                f"Gap in validation windows: expected {written_until}, "
                f"next window starts at {start}."
            )

        context_len = written_until - start

        # Ignore redundant trailing windows.
        if context_len >= valid_length:
            continue

        if context_len > context_frames:
            raise RuntimeError(
                "Window overlap exceeds configured context_frames. "
                "Check stride_frames."
            )

        # =====================================
        # PREVIOUS ACTION CONTEXT
        # =====================================

        if start == 0:
            prefix_actions = None

        else:
            # Previously generated actions, plus the preceding action
            # required by the one-frame teacher-forcing alignment.
            prefix_actions = generated[
                start - 1 : written_until
            ].unsqueeze(0)

            if prefix_actions.size(1) != context_len + 1:
                raise RuntimeError("Previous action context is incomplete.")

        # =====================================
        # GLOBAL CONTEXT
        # =====================================

        generate_kwargs = {}

        if full_global_context is not None:

            global_context = full_global_context[
                :,
                start : start + seq_len,
                :
            ]

            # Pad the final window if necessary.
            if global_context.size(1) < seq_len:
                global_context = torch.nn.functional.pad(
                    global_context,
                    (0, 0, 0, seq_len - global_context.size(1))
                )

            generate_kwargs["global_context"] = global_context

        # =====================================
        # GENERATE
        # =====================================

        results = trainer.generate(
            seg["sample"],
            seed_action=seed_action,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            seed=seed if window_idx == 0 else None,
            prefix_actions=prefix_actions,
            **generate_kwargs,
        )

        predictions = results["action_predictions"][0]

        # Remove predictions belonging to the known overlap.
        new_predictions = predictions[
            context_len:valid_length
        ]

        actual_new = seg["actual"][
            context_len:valid_length
        ]

        expected_new = end - written_until

        if new_predictions.numel() != expected_new:
            raise RuntimeError("Unexpected number of new predictions.")

        # Write only newly generated positions.
        generated[written_until:end] = new_predictions
        actual[written_until:end] = actual_new

        # =====================================
        # DEBUG
        # =====================================

        if collect_debug:

            debug_segments.append({
                "start": written_until + 1,
                "window_start": start + 1,
                "end": end + 1,
                "action_inputs": results["action_inputs"][0][
                    context_len:valid_length
                ],
                "action_predictions": new_predictions,
                "np_pred": results["np_pred"][0][
                    context_len:valid_length
                ],
                "ns_logits": results["ns_logits"][0][
                    context_len:valid_length
                ],
                "actual": actual_new,
                "valid_frames": results["valid_frames"][0][
                    context_len:valid_length
                ],
            })

        print(
            f"Window {start:6d}-{end:6d} | "
            f"Context: {context_len:4d} | "
            f"New: {len(new_predictions):4d} | "
            f"Generated active: {(new_predictions != 0).sum().item():4d} | "
            f"Actual active: {(actual_new != 0).sum().item():4d}"
        )

        written_until = end
        window_idx += 1

    if written_until != total_frames:
        raise RuntimeError(
            f"Chart generation incomplete: "
            f"{written_until}/{total_frames} frames."
        )

    # =====================================
    # SUMMARY
    # =====================================

    print(f"\nDone. {len(generated)} frames.")

    print(
        f"Active frames: generated {(generated != 0).sum().item()} "
        f"| actual {(actual != 0).sum().item()}"
    )

    return generated, actual, debug_segments
