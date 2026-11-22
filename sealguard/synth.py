"""Synthetic blister-strip images, for testing the pipeline before real photos exist.

These are drawings, not photographs. Numbers measured on them say the code
works; they say nothing about accuracy on real strips.
"""
import cv2
import numpy as np

DEFECTS = ["empty", "crack", "chip", "foreign", "wrong_colour"]
ROWS, COLS = 2, 5
SW, SH = 600, 240


def _tablet(canvas, cx, cy, r, colour):
    yy, xx = np.mgrid[0:canvas.shape[0], 0:canvas.shape[1]]
    d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    shade = np.clip(1.0 - 0.18 * (d / r) ** 2, 0, 1)[..., None]
    mask = (d <= r)[..., None]
    canvas[:] = np.where(mask, np.array(colour, np.float32) * shade, canvas)


def make_strip(rng, defects=None):
    """defects: dict {pocket_index (0-based): defect name}. Returns a BGR camera frame."""
    defects = defects or {}
    strip = np.full((SH, SW, 3), (188, 190, 193), np.float32)
    strip += cv2.GaussianBlur(rng.normal(0, 14, (SH, SW, 1)).astype(np.float32), (0, 0), 3)[..., None]
    cw, ch = (SW - 48) / COLS, (SH - 19) / ROWS
    for i in range(ROWS * COLS):
        r_, c_ = divmod(i, COLS)
        cx = 24 + (c_ + 0.5) * cw + rng.normal(0, 1.2)
        cy = 9.5 + (r_ + 0.5) * ch + rng.normal(0, 1.2)
        cv2.circle(strip, (int(cx), int(cy)), 47, (206, 207, 209), -1, cv2.LINE_AA)   # blister dome
        cv2.circle(strip, (int(cx), int(cy)), 47, (150, 152, 155), 1, cv2.LINE_AA)
        kind = defects.get(i)
        if kind == "empty":
            continue
        colour = (236, 240, 242)
        if kind == "wrong_colour":
            colour = [(150, 190, 240), (200, 170, 235), (170, 225, 200)][rng.integers(3)]
        _tablet(strip, cx, cy, 36 + rng.normal(0, 0.4), colour)
        if kind == "crack":
            a = rng.uniform(0, np.pi)
            pts = [(cx + t * 30 * np.cos(a) + rng.normal(0, 2.5), cy + t * 30 * np.sin(a) + rng.normal(0, 2.5))
                   for t in np.linspace(-1, 1, 6)]
            cv2.polylines(strip, [np.int32(pts)], False, (95, 98, 100), 2, cv2.LINE_AA)
        elif kind == "chip":
            a = rng.uniform(0, 2 * np.pi)
            cv2.circle(strip, (int(cx + 36 * np.cos(a)), int(cy + 36 * np.sin(a))),
                       int(rng.integers(11, 17)), (168, 170, 174), -1, cv2.LINE_AA)  # shadowed gap
        elif kind == "foreign":
            a, rr = rng.uniform(0, 2 * np.pi), rng.uniform(0, 40)
            cv2.circle(strip, (int(cx + rr * np.cos(a)), int(cy + rr * np.sin(a))),
                       int(rng.integers(4, 7)), (40, 45, 60), -1, cv2.LINE_AA)
    # place the strip on a dark base with small rotation, shift and lighting changes
    frame = np.full((480, 720, 3), 24, np.float32)
    x0, y0 = 60 + int(rng.integers(-15, 16)), 120 + int(rng.integers(-15, 16))
    frame[y0:y0 + SH, x0:x0 + SW] = strip
    M = cv2.getRotationMatrix2D((360, 240), float(rng.uniform(-3, 3)), 1.0)
    frame = cv2.warpAffine(frame, M, (720, 480), borderValue=(24, 24, 24))
    frame = frame * rng.uniform(0.96, 1.04) + rng.normal(0, 2.5, frame.shape)
    return np.clip(frame, 0, 255).astype(np.uint8)
