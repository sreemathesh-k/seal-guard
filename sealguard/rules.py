"""Fast colour rules that run before the anomaly model: empty pocket, wrong colour."""
import cv2
import numpy as np


def _lab(crop):
    return cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).astype(np.float32)


def centre_colour(crop):
    lab = _lab(crop)
    n = crop.shape[0]
    a, b = int(0.35 * n), int(0.65 * n)
    return np.median(lab[a:b, a:b].reshape(-1, 3), 0)  # median ignores thin cracks and specks


def corner_colour(crop):
    """Colour of the foil in the four corners of the crop (outside the blister)."""
    lab = _lab(crop)
    k = max(4, crop.shape[0] // 16)
    patches = [lab[:k, :k], lab[:k, -k:], lab[-k:, :k], lab[-k:, -k:]]
    return np.concatenate([p.reshape(-1, 3) for p in patches]).mean(0)


def fit_rules(good_crops):
    centres = np.array([centre_colour(c) for c in good_crops])
    ref = np.median(centres, 0)
    de = np.linalg.norm(centres - ref, axis=1)
    return {
        "tablet_lab": ref.tolist(),
        "foil_lab": np.median([corner_colour(c) for c in good_crops], 0).tolist(),
        "de_thr": float(max(12.0, de.mean() + 6 * de.std())),
    }


def apply_rules(crop, rules):
    """Return 'empty', 'wrong_colour' or None."""
    c = centre_colour(crop)
    d_tab = np.linalg.norm(c - np.array(rules["tablet_lab"]))
    if d_tab <= rules["de_thr"]:
        return None
    foil = np.array(rules["foil_lab"])
    # an empty pocket shows the foil; a wrong tablet has its own colour
    gap = np.linalg.norm(np.array(rules["tablet_lab"]) - foil)
    return "empty" if np.linalg.norm(c - foil) < 0.5 * gap else "wrong_colour"
