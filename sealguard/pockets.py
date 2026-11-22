"""Find the blister strip in a camera frame, straighten it, and cut out each pocket.

Assumptions (match the light-box rig):
  * the strip lies on a dark matte base, so it is the largest bright object;
  * it is fed in a fixed orientation, rotated less than 45 degrees;
  * pockets sit on a regular rows x cols grid.
"""
import cv2
import numpy as np

STRIP_WH = (600, 240)   # straightened strip size in pixels (width, height)
POCKET = 128            # pocket crop size fed to the models
TARGET_GRAY = 170.0     # strip brightness is normalised to this median


def find_strip(img, out_wh=STRIP_WH):
    """Return the straightened strip image, or None if no strip is visible."""
    gray = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(c) < 0.05 * img.shape[0] * img.shape[1]:
        return None
    box = cv2.boxPoints(cv2.minAreaRect(c)).astype(np.float32)
    s, d = box.sum(1), box[:, 1] - box[:, 0]
    src = np.float32([box[s.argmin()], box[d.argmin()], box[s.argmax()], box[d.argmax()]])  # tl,tr,br,bl
    w, h = out_wh
    dst = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
    strip = cv2.warpPerspective(img, cv2.getPerspectiveTransform(src, dst), (w, h))
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


def crop_pockets(strip, rows, cols, size=POCKET):
    """Return (list of size x size BGR crops, list of boxes). Pocket 1 is top-left."""
    boxes = pocket_boxes(rows, cols, (strip.shape[1], strip.shape[0]))
    crops = [cv2.resize(strip[y0:y1, x0:x1], (size, size), interpolation=cv2.INTER_AREA)
             for x0, y0, x1, y1 in boxes]
    return crops, boxes


def strip_to_pockets(img, rows, cols):
    strip = find_strip(img)
    if strip is None:
        return None, [], []
    crops, boxes = crop_pockets(strip, rows, cols)
    return strip, crops, boxes
