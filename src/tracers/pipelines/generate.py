from __future__ import annotations


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
