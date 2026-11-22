"""PatchCore anomaly model in plain NumPy (Roth et al., CVPR 2022).

Keeps a small "memory bank" of patch features from GOOD images, chosen by
greedy coreset sampling so it covers the variety of normal patches. A test
patch's anomaly score is the distance to its nearest neighbour in the bank.

Same interface as Padim (fit / score_maps / scores / set_threshold / save /
load), and NumPy-only so it runs unchanged on the Raspberry Pi.
"""
import cv2
import numpy as np


def _to_rows(feats):
    n, c, h, w = feats.shape
    return np.ascontiguousarray(feats.transpose(0, 2, 3, 1).reshape(-1, c), dtype=np.float32)


def greedy_coreset(x, m, proj_dim=128, seed=0):
    """Pick m rows of x that cover the whole set (k-centre greedy, on a random projection)."""
    rng = np.random.default_rng(seed)
    if x.shape[1] > proj_dim:
        p = rng.normal(size=(x.shape[1], proj_dim)).astype(np.float32) / np.sqrt(proj_dim)
        z = x @ p
    else:
        z = x
    sq = (z * z).sum(1)
    first = int(rng.integers(len(z)))
    sel = [first]
    d = np.maximum(sq - 2 * (z @ z[first]) + sq[first], 0)
    for _ in range(m - 1):
        i = int(d.argmax())
        sel.append(i)
        d = np.minimum(d, np.maximum(sq - 2 * (z @ z[i]) + sq[i], 0))
    return np.array(sel)


class PatchCore:
    def __init__(self, ratio=0.01, min_bank=1000, proj_dim=128, seed=0):
        self.ratio, self.min_bank, self.proj_dim, self.seed = ratio, min_bank, proj_dim, seed
        self.bank = self.hw = self.thr = None

    def fit(self, feats):
        """feats: (N, C, H, W) from good images/pockets."""
        x = _to_rows(feats)
        m = min(len(x), max(self.min_bank, int(self.ratio * len(x))))
        self.bank = x[greedy_coreset(x, m, self.proj_dim, self.seed)]
        self.hw = feats.shape[2:]
        return self

    def score_maps(self, feats, chunk=4096):
        n, _, h, w = feats.shape
        q = _to_rows(feats)
        bank_t = np.ascontiguousarray(self.bank.T)
        bsq = (self.bank * self.bank).sum(1)
        out = np.empty(len(q), np.float32)
        for i in range(0, len(q), chunk):
            b = q[i:i + chunk]
            d = (b * b).sum(1)[:, None] - 2 * (b @ bank_t) + bsq[None]
            out[i:i + chunk] = np.sqrt(np.maximum(d.min(1), 0))
        maps = out.reshape(n, h, w)
        return np.stack([cv2.GaussianBlur(a, (5, 5), 1.0) for a in maps])

    @staticmethod
    def scores(maps):
        """Image/pocket score = mean of the 10 highest patches."""
        flat = np.sort(maps.reshape(len(maps), -1), axis=1)
        return flat[:, -10:].mean(1)

    def set_threshold(self, good_scores, k=6.0):
        s = np.asarray(good_scores, np.float64)
        self.thr = float(max(s.mean() + k * s.std(), s.max() * 1.3))
        return self.thr

    def save(self, path):
        np.savez_compressed(path, bank=self.bank, hw=np.array(self.hw), thr=np.array(self.thr))

    @classmethod
    def load(cls, path):
        z = np.load(path)
        m = cls()
        m.bank, m.hw, m.thr = z["bank"], tuple(int(v) for v in z["hw"]), float(z["thr"])
        return m
