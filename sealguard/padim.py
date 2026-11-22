"""PaDiM-style anomaly model in plain NumPy.

Learns, for every patch position of a pocket, the mean and covariance of the
backbone features seen on GOOD pockets only. At inspection time the
Mahalanobis distance of each patch to that "normal" distribution is the
anomaly score. No defect images are needed.

Keeping this part in NumPy means the only neural network to deploy is the
backbone (one .pte file); this file runs unchanged on the Raspberry Pi.
"""
import cv2
import numpy as np


class Padim:
    def __init__(self, d=100, eps=0.01, seed=0):
        self.d, self.eps, self.seed = d, eps, seed
        self.idx = self.mean = self.inv = None
        self.thr = None

    # ---- training on good pockets -------------------------------------
    def fit(self, feats):
        """feats: (N, C, H, W) float array from good pockets."""
        n, c, h, w = feats.shape
        rng = np.random.default_rng(self.seed)
        self.idx = np.sort(rng.choice(c, self.d, replace=False)) if c > self.d else np.arange(c)
        x = feats[:, self.idx].reshape(n, len(self.idx), h * w).transpose(2, 0, 1).astype(np.float32)  # P,N,d
        self.mean = x.mean(1)
        xc = x - self.mean[:, None]
        cov = (xc.transpose(0, 2, 1) @ xc).astype(np.float64) / max(n - 1, 1)
        # regularise relative to the average feature variance so it works for any backbone scale
        lam = self.eps * float(np.mean(np.einsum("pcc->pc", cov))) + 1e-8
        cov += lam * np.eye(len(self.idx))[None]
        self.inv = np.linalg.inv(cov).astype(np.float32)
        self.mean = self.mean.astype(np.float32)
        self.hw = (h, w)
        return self

    def set_threshold(self, good_scores, k=6.0):
        """Set the pass/fail threshold from scores of held-out GOOD pockets."""
        s = np.asarray(good_scores, np.float64)
        # k sigmas above the mean, and at least 30 % above the worst good pocket seen
        self.thr = float(max(s.mean() + k * s.std(), s.max() * 1.3))
        return self.thr

    # ---- inspection -----------------------------------------------------
    def score_maps(self, feats):
        """Return anomaly maps (N, H, W)."""
        n = feats.shape[0]
        h, w = self.hw
        x = feats[:, self.idx].reshape(n, len(self.idx), h * w).transpose(0, 2, 1).astype(np.float32)
        diff = x - self.mean[None]
        t = np.einsum("npc,pcd->npd", diff, self.inv)
        m = np.sqrt(np.maximum((t * diff).sum(-1), 0)).reshape(n, h, w)
        return np.stack([cv2.GaussianBlur(a, (5, 5), 1.0) for a in m])

    @staticmethod
    def scores(maps):
        """Pocket score = mean of the 10 highest patches (steadier than a single max)."""
        flat = np.sort(maps.reshape(len(maps), -1), axis=1)
        return flat[:, -10:].mean(1)

    # ---- persistence ----------------------------------------------------
    def save(self, path):
        np.savez_compressed(path, idx=self.idx, mean=self.mean, inv=self.inv,
                            hw=np.array(self.hw), thr=np.array(self.thr))

    @classmethod
    def load(cls, path):
        z = np.load(path)
        m = cls()
        m.idx, m.mean, m.inv = z["idx"], z["mean"], z["inv"]
        m.hw, m.thr = tuple(int(v) for v in z["hw"]), float(z["thr"])
        return m
