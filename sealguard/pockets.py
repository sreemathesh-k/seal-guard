"""Find the blister strip in a camera frame, straighten it, and cut out each pocket.

Assumptions (match the light-box rig):
  * the strip lies on a dark matte base, so it is the largest bright object;
  * any rotation; the long side is turned horizontal (a 180-degree flip is not resolved);
  * pockets sit on a regular rows x cols grid.
"""
import cv2
import numpy as np

STRIP_WH = (600, 240)   # straightened strip size in pixels (width, height)
POCKET = 128            # pocket crop size fed to the models
TARGET_GRAY = 170.0     # strip brightness is normalised to this median


def find_strip(img, out_wh=STRIP_WH):
    """Return the straightened strip image (long side horizontal), or None."""
    sc = min(1.0, 800.0 / max(img.shape[:2]))
    sm = cv2.resize(img, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA) if sc < 1 else img
    v = cv2.GaussianBlur(cv2.cvtColor(sm, cv2.COLOR_BGR2HSV)[..., 2], (5, 5), 0)
    _, mask = cv2.threshold(v, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((15, 15), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(c) < 0.03 * mask.size:
        return None
    box = cv2.boxPoints(cv2.minAreaRect(c)) / sc
    m = box.mean(0)
    box = box[np.argsort(np.arctan2(box[:, 1] - m[1], box[:, 0] - m[0]))]
    if np.linalg.norm(box[0] - box[1]) < np.linalg.norm(box[1] - box[2]):
        box = np.roll(box, -1, axis=0)          # long side -> output width
    if box[2].sum() < box[0].sum():
        box = np.roll(box, 2, axis=0)           # start nearest the image top-left
    w, h = out_wh
    dst = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
    strip = cv2.warpPerspective(img, cv2.getPerspectiveTransform(box.astype(np.float32), dst), (w, h))
    # one gain for all channels: removes lamp/exposure drift but keeps colour
    gain = TARGET_GRAY / max(float(np.median(cv2.cvtColor(strip, cv2.COLOR_BGR2GRAY))), 1.0)
    return np.clip(strip.astype(np.float32) * gain, 0, 255).astype(np.uint8)


def pocket_boxes(rows, cols, strip_wh=STRIP_WH, margin=0.04):
    """Grid boxes (x0, y0, x1, y1) on the straightened strip, row by row."""
    w, h = strip_wh
    mx, my = margin * w, margin * h
    cw, ch = (w - 2 * mx) / cols, (h - 2 * my) / rows
    return [(int(mx + c * cw), int(my + r * ch), int(mx + (c + 1) * cw), int(my + (r + 1) * ch))
            for r in range(rows) for c in range(cols)]


def _recentre(gray, box):
    """Move a grid cell onto the round blister inside it (Hough circle); keep the grid if none."""
    x0, y0, x1, y1 = box
    H, W = gray.shape
    cw, ch = x1 - x0, y1 - y0
    cx, cy, r0 = (x0 + x1) / 2, (y0 + y1) / 2, min(cw, ch) / 2
    pad = int(0.3 * r0)
    wx, wy = max(0, x0 - pad), max(0, y0 - pad)
    win = gray[wy:min(H, y1 + pad), wx:min(W, x1 + pad)]
    c = cv2.HoughCircles(win, cv2.HOUGH_GRADIENT, dp=1.2, minDist=r0, param1=120, param2=20,
                         minRadius=int(0.6 * r0), maxRadius=int(1.05 * r0))
    if c is not None:
        q = min(c[0], key=lambda q: (q[0] + wx - cx) ** 2 + (q[1] + wy - cy) ** 2)
        if abs(q[0] + wx - cx) < 0.35 * cw and abs(q[1] + wy - cy) < 0.35 * ch:
            cx, cy = q[0] + wx, q[1] + wy
    return (int(max(0, cx - cw / 2)), int(max(0, cy - ch / 2)),
            int(min(W, cx + cw / 2)), int(min(H, cy + ch / 2)))


def crop_pockets(strip, rows, cols, size=POCKET, recentre=True):
    """Return (list of size x size BGR crops, list of boxes). Pocket 1 is top-left."""
    boxes = pocket_boxes(rows, cols, (strip.shape[1], strip.shape[0]))
    if recentre:
        gray = cv2.medianBlur(cv2.cvtColor(strip, cv2.COLOR_BGR2GRAY), 5)
        boxes = [_recentre(gray, b) for b in boxes]
    crops = [cv2.resize(strip[y0:y1, x0:x1], (size, size), interpolation=cv2.INTER_AREA)
             for x0, y0, x1, y1 in boxes]
    return crops, boxes


def strip_to_pockets(img, rows, cols):
    strip = find_strip(img)
    if strip is None:
        return None, [], []
    crops, boxes = crop_pockets(strip, rows, cols)
    return strip, crops, boxes
