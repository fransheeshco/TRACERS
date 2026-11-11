from __future__ import annotations
from pathlib import Path
from tracers.export.chart import export_generated_chart
from tracers.generate.converter import export_generation_debug

def run_validation_generation(trainer, loader, *, chart_id, context_frames, **kwargs):
    from tracers.generate.converter import generate_validation_chart

    generated, actual, debug_segments = generate_validation_chart(
        trainer,
        loader,
        chart_id=chart_id,
        context_frames=context_frames,
        **kwargs,
    )
    return generated, actual, debug_segments


def run_custom_generation(trainer, audio_pt, *, difficulty, segment_frames, **kwargs):
    from tracers.generate.converter import generate_custom_song

    generated, debug_segments = generate_custom_song(
        trainer,
        audio_pt,
        difficulty=difficulty,
        segment_frames=segment_frames,
        **kwargs,
    )
    return generated, debug_segments

def run_generation_sweep(
    trainer,
    audio_path,
    difficulty,
    segment_frames,
    configs,
    bpm=120,
    offset_ms=0,
    lane_map=(0, 2, 4, 6),
    seed_action=0,
    default_seed=0,
    output_root="charts",
    save_debug_csv=True,
    debug_top_k=5,
    start_frame=0,
):
    """
    Generate and export multiple configurations for one song.

    Each config can contain:
        name
        context_frames
        temperature
        top_k
        top_p
        seed

    Returns a list of run summaries, including generated action tensors.
    """

    audio_path = Path(audio_path)
    audio_name = audio_path.stem

    output_dir = Path(output_root) / audio_name
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []

    def format_tag(value):
        if value is None:
            return "none"
        return str(value).replace(".", "p")

    print(f"Generating sweep for: {audio_name}")
    print(f"Number of configurations: {len(configs)}")
    print(f"Output directory: {output_dir}")

    for idx, config in enumerate(configs, start=1):

        config = dict(config)

        # =====================================
        # CONFIGURATION
        # =====================================

        name = config.pop("name", f"run{idx:02d}")

        context_frames = config.pop("context_frames", 0)
        temperature = config.pop("temperature", None)
        top_k = config.pop("top_k", None)
        top_p = config.pop("top_p", None)
        seed = config.pop("seed", default_seed)

        if config:
            raise ValueError(
                f"Unknown configuration arguments: {list(config.keys())}"
            )

        if not 0 <= context_frames < segment_frames:
            raise ValueError(
                f"Invalid context_frames={context_frames}. "
                f"Must be smaller than segment_frames={segment_frames}."
            )

        # =====================================
        # GENERATE
        # =====================================

        print(f"\n{'=' * 60}")
        print(f"RUN {idx}/{len(configs)}: {name}")
        print(f"{'=' * 60}")

        print(
            f"Context: {context_frames} | "
            f"Temperature: {temperature} | "
            f"Top-K: {top_k} | "
            f"Top-P: {top_p} | "
            f"Seed: {seed}"
        )

        generated, debug_segments = run_custom_generation(
            trainer,
            audio_path,
            difficulty=difficulty,
            segment_frames=segment_frames,
            context_frames=context_frames,
            seed_action=seed_action,
            seed=seed,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            collect_debug=save_debug_csv,
        )

        # =====================================
        # FILE NAMING
        # =====================================

        run_meta = (
            f"{idx:02d}_{name}"
            f"_diff{difficulty}"
            f"_ctx{context_frames}"
            f"_temp{format_tag(temperature)}"
            f"_topk{format_tag(top_k)}"
            f"_topp{format_tag(top_p)}"
            f"_seed{seed}"
        )

        chart_path = output_dir / f"chart_{run_meta}.json"
        debug_path = output_dir / f"debug_{run_meta}.csv"

        # =====================================
        # EXPORT CHART
        # =====================================

        export_generated_chart(
            generated,
            chart_path,
            bpm=bpm,
            start_frame=start_frame,
            offset_ms=offset_ms,
            lane_map=lane_map,
        )

        print(f"Saved chart: {chart_path}")

        # =====================================
        # EXPORT DEBUG
        # =====================================

        if save_debug_csv and debug_segments:
            export_generation_debug(
                debug_segments,
                output_path=str(debug_path),
                top_k=debug_top_k,
            )

            print(f"Saved debug: {debug_path}")

        # =====================================
        # SUMMARY
        # =====================================

        active_frames = int((generated != 0).sum().item())
        total_frames = len(generated)
        active_rate = active_frames / max(total_frames, 1)

        print(
            f"Result: {active_frames}/{total_frames} active frames "
            f"({active_rate:.2%})"
        )

        results.append({
            "name": name,
            "config": {
                "context_frames": context_frames,
                "temperature": temperature,
                "top_k": top_k,
                "top_p": top_p,
                "seed": seed,
            },
            "generated": generated.cpu(),
            "active_frames": active_frames,
            "active_rate": active_rate,
            "chart_path": chart_path,
            "debug_path": debug_path if save_debug_csv else None,
        })

    # =====================================
    # SWEEP SUMMARY
    # =====================================

    print(f"\n{'=' * 60}")
    print("GENERATION SWEEP SUMMARY")
    print(f"{'=' * 60}")

    for result in results:
        print(
            f"{result['name']:20s} | "
            f"Active: {result['active_frames']:5d} | "
            f"Rate: {result['active_rate']:7.2%}"
        )

    print(f"\nAll files saved to: {output_dir}")

    return results
