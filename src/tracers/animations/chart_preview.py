"""Render a 4-lane chart preview from generated action tokens, using your own note sprites.

Pure Pillow, no Gradio dependency. Typical use while generating:

    assets = NoteAssets("assets/notes_sheet.png", "assets/hold_body.png")
    img = render_progress(all_actions_so_far, assets, total_frames=total_frames)

Token format: base-4 per lane, lane i = (token // 4**i) % 4
    0 = empty, 1 = tap, 2 = hold start, 3 = hold end
One token per 10 ms frame.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional

from PIL import Image, ImageDraw, ImageFont

FRAME_MS = 10

# Sprite sheet grid (your 2048x1024 sheet): 310 px columns, 122 px rows, first row at y=50.
CELL_W, CELL_H = 310, 122
ROW_Y0, ROW_PITCH = 50, 122


@dataclass
class Note:
    lane: int
    t0_ms: float
    t1_ms: float
    kind: str  # "tap" | "hold" | "open" (hold still being generated) | "orphan" (invalid grammar)


def parse_notes(actions: Iterable[int], first_frame: int = 1) -> List[Note]:
    """Turn frame tokens into notes. Re-parse the full sequence each redraw; it is cheap."""
    notes: List[Note] = []
    open_holds: dict[int, float] = {}

    for i, tok in enumerate(actions):
        tok = int(tok)
        if tok == 0:
            continue
        t = (first_frame + i) * FRAME_MS
        for lane in range(4):
            state = (tok // (4 ** lane)) % 4
            if state == 1:
                notes.append(Note(lane, t, t, "tap"))
            elif state == 2:
                if lane in open_holds:  # new start while one is open: previous start is orphaned
                    notes.append(Note(lane, open_holds[lane], open_holds[lane], "orphan"))
                open_holds[lane] = t
            elif state == 3:
                if lane in open_holds:
                    notes.append(Note(lane, open_holds.pop(lane), t, "hold"))
                else:  # end without a start
                    notes.append(Note(lane, t, t, "orphan"))

    for lane, t0 in open_holds.items():
        notes.append(Note(lane, t0, t0, "open"))  # t1 is set to the generation front when drawing
    return notes


class NoteAssets:
    """Loads and caches your sprites.

    tap / hold are (row, col) cells in the sheet. Hold head and hold end use the same sprite.
    The hold body is stretched from a thin horizontal strip of your body texture.
    """

    def __init__(self, sheet_path, body_path, tap=(7, 5), hold=(0, 1), body_opacity_boost=2.3):
        sheet = Image.open(sheet_path).convert("RGBA")
        self._tap = self._cell(sheet, *tap)
        self._hold = self._cell(sheet, *hold)

        body = Image.open(body_path).convert("RGBA")
        w, h = body.size
        strip = body.crop((0, h // 2 - 8, w, h // 2 + 8))
        # Your body texture is ~38% opaque (made to sit over a game lane); boost it for a dark background.
        r, g, b, a = strip.split()
        lut = [min(235, int(v * body_opacity_boost)) for v in range(256)]
        a = a.point(lut)
        strip.putalpha(a)
        self._body_strip = strip
        self._cache: dict = {}

    @staticmethod
    def _cell(sheet: Image.Image, row: int, col: int) -> Image.Image:
        x0 = max(0, col * CELL_W - 2)
        y0 = ROW_Y0 + row * ROW_PITCH
        cell = sheet.crop((x0, y0, min(sheet.width, x0 + CELL_W + 2), y0 + CELL_H))
        bbox = cell.getbbox()
        return cell.crop(bbox) if bbox else cell

    def sprite(self, name: str, width: int) -> Image.Image:
        key = (name, width)
        if key not in self._cache:
            src = self._tap if name == "tap" else self._hold
            h = max(1, round(src.height * width / src.width))
            self._cache[key] = src.resize((width, h), Image.Resampling.LANCZOS)
        return self._cache[key]

    def body(self, width: int, height: int) -> Image.Image:
        return self._body_strip.resize((max(1, width), max(1, height)), Image.Resampling.BILINEAR)


def _font(size: int):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # older Pillow
        return ImageFont.load_default()


def _paste(layer: Image.Image, sprite: Image.Image, x: int, y: int) -> None:
    layer.paste(sprite, (x, y), sprite)


def _render_column(notes, t_lo, t_hi, front_ms, assets, *, lane_w, px_per_s,
                   hold_w_ratio, note_w_ratio, axis_w=44, tick_ms=2000):
    """One vertical strip of the chart covering [t_lo, t_hi] ms. Older at the bottom, newer on top."""
    h = int((t_hi - t_lo) / 1000.0 * px_per_s)
    W = axis_w + 4 * lane_w + 10

    def y_of(t_ms):
        return (t_hi - t_ms) / 1000.0 * px_per_s

    bg = Image.new("RGBA", (W, h), (16, 18, 28, 255))
    d = ImageDraw.Draw(bg)
    for i in range(4):
        x0 = axis_w + i * lane_w
        shade = (24, 27, 40, 255) if i % 2 == 0 else (20, 23, 35, 255)
        d.rectangle([x0, 0, x0 + lane_w - 1, h], fill=shade)
    for i in range(5):
        x = axis_w + i * lane_w
        d.line([(x, 0), (x, h)], fill=(52, 58, 80, 255), width=1)

    font_s = _font(12)
    t = int(-(-t_lo // tick_ms)) * tick_ms
    while t <= t_hi:
        y = y_of(t)
        d.line([(axis_w - 6, y), (axis_w, y)], fill=(90, 98, 130, 255))
        d.text((4, y - 6), f"{t // 60000}:{(t // 1000) % 60:02d}", fill=(130, 138, 170, 255), font=font_s)
        t += tick_ms

    bodies = Image.new("RGBA", (W, h), (0, 0, 0, 0))
    heads = Image.new("RGBA", (W, h), (0, 0, 0, 0))
    taps = Image.new("RGBA", (W, h), (0, 0, 0, 0))

    note_w = int(lane_w * note_w_ratio)
    body_w = int(lane_w * hold_w_ratio)
    hold_sprite = assets.sprite("hold", note_w)
    tap_sprite = assets.sprite("tap", note_w)
    margin = 25

    for n in notes:
        if n.t1_ms < t_lo - 400 or n.t0_ms > t_hi + 400:
            continue
        lane_x = axis_w + n.lane * lane_w

        if n.kind in ("hold", "open"):
            y_top, y_bot = max(y_of(n.t1_ms), 0), min(y_of(n.t0_ms), h)
            if y_bot > y_top:
                _paste(bodies, assets.body(body_w, int(y_bot - y_top)),
                       lane_x + (lane_w - body_w) // 2, int(y_top))

        if n.kind == "tap":
            y = y_of(n.t0_ms)
            if -margin <= y <= h + margin:
                _paste(taps, tap_sprite, lane_x + (lane_w - note_w) // 2, int(y - tap_sprite.height / 2))
        else:  # hold head, hold end, and orphans all use the same green note
            ys = [y_of(n.t0_ms)] if n.kind in ("open", "orphan") else [y_of(n.t0_ms), y_of(n.t1_ms)]
            for y in ys:
                if -margin <= y <= h + margin:
                    _paste(heads, hold_sprite, lane_x + (lane_w - note_w) // 2, int(y - hold_sprite.height / 2))

    col = Image.alpha_composite(bg, bodies)
    col = Image.alpha_composite(col, heads)
    col = Image.alpha_composite(col, taps)

    # Region that has not been generated yet: dim it. No notes exist there, so nothing is revealed early.
    if front_ms < t_hi:
        y_front = max(0.0, min(h, y_of(front_ms)))
        shade = Image.new("RGBA", (W, h), (0, 0, 0, 0))
        ImageDraw.Draw(shade).rectangle([axis_w, 0, axis_w + 4 * lane_w, y_front], fill=(8, 9, 14, 170))
        col = Image.alpha_composite(col, shade)
        if t_lo <= front_ms <= t_hi:
            ImageDraw.Draw(col).line([(axis_w, y_front), (axis_w + 4 * lane_w, y_front)],
                                     fill=(96, 220, 120, 255), width=2)
    return col.convert("RGB")


def render_progress(
    actions: Iterable[int],
    assets: NoteAssets,
    *,
    total_frames: Optional[int] = None,
    first_frame: int = 1,
    view_seconds: float = 20.0,
    columns: int = 3,
    lane_w: int = 90,
    px_per_s: float = 90.0,
    hold_w_ratio: float = 0.78,
    note_w_ratio: float = 0.92,
    col_gap: int = 12,
) -> Image.Image:
    """Draw the most recent `view_seconds` of the chart generated so far, split into columns.

    Reading order is left to right; inside each column older notes are at the bottom and newer
    ones on top (like notes falling toward a judgment line). Only generated notes are drawn.
    """
    actions = list(actions)
    front_ms = (first_frame + len(actions)) * FRAME_MS
    notes = parse_notes(actions, first_frame)
    for n in notes:
        if n.kind == "open":
            n.t1_ms = front_ms

    view_ms = view_seconds * 1000.0
    t_lo = max(0.0, front_ms - view_ms)
    seg = view_ms / columns

    cols = [
        _render_column(notes, t_lo + i * seg, t_lo + (i + 1) * seg, front_ms, assets,
                       lane_w=lane_w, px_per_s=px_per_s,
                       hold_w_ratio=hold_w_ratio, note_w_ratio=note_w_ratio)
        for i in range(columns)
    ]

    head_h = 58
    W = sum(c.width for c in cols) + col_gap * (columns - 1)
    H = head_h + cols[0].height + 10
    out = Image.new("RGB", (W, H), (12, 13, 20))
    x = 0
    for c in cols:
        out.paste(c, (x, head_h))
        x += c.width + col_gap

    d = ImageDraw.Draw(out)
    font_h, font_s = _font(16), _font(13)
    front_s = front_ms / 1000.0
    bar_x0, bar_x1 = 12, W - 12
    if total_frames:
        total_s = (first_frame + total_frames) * FRAME_MS / 1000.0
        frac = min(1.0, front_s / total_s)
        label = f"Generating...  {front_s:5.1f}s / {total_s:.1f}s  ({frac * 100:.0f}%)"
        d.rectangle([bar_x0, 38, bar_x1, 46], fill=(40, 44, 64))
        d.rectangle([bar_x0, 38, bar_x0 + int((bar_x1 - bar_x0) * frac), 46], fill=(96, 220, 120))
    else:
        label = f"Generating...  {front_s:.1f}s"
    d.text((bar_x0, 12), label, fill=(225, 230, 245), font=font_h)
    n_notes = sum(1 for n in notes if n.kind in ("tap", "hold", "open"))
    d.text((bar_x1 - 70, 14), f"{n_notes} notes", fill=(150, 158, 190), font=font_s)
    return out
