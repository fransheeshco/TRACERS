"""Parsing helpers for Bestdori/Bandori chart JSON."""

from tracers.audio.beat_features import TempoChange


def tempo_changes_from_bestdori(chart: object) -> list[TempoChange]:
    """Read and validate all BPM events from a Bestdori chart."""

    if not isinstance(chart, list):
        raise ValueError("Bestdori chart JSON must contain a top-level array")

    changes: list[TempoChange] = []
    for event in chart:
        if isinstance(event, dict) and event.get("type") == "BPM":
            try:
                changes.append(TempoChange(beat=float(event["beat"]), bpm=float(event["bpm"])))
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"Invalid BPM event: {event!r}") from error

    if not changes:
        raise ValueError("Chart contains no BPM events")
    return changes
