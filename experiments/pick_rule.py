import numpy as np, itertools
a = np.load("/tmp/feat_train.npy"); e = a[:, 1] == 1; g = ~e
vs, ed, vm = a[:, 4], a[:, 5], a[:, 2]
print("good n", g.sum(), "empty n", e.sum())
for name, x in [("V_std", vs), ("edges", ed)]:
    print(name, "good sorted top5:", np.round(np.sort(x[g])[-5:], 1))
    for t in sorted(set(np.round(np.percentile(x[e], [10, 25, 40, 50, 60, 75]), 1))):
        print(f"  {name}>{t:5.1f}: empties caught {int((x[e] > t).sum())}/{e.sum()}, goods flagged {int((x[g] > t).sum())}/{g.sum()}")
print("V_mean<250: empties", int((vm[e] < 250).sum()), "goods", int((vm[g] < 250).sum()))
best = None
for t1, t2 in itertools.product(np.arange(2, 12, 0.5), np.arange(2, 18, 0.5)):
    f = (vs > t1) | (ed > t2)
    fp, tp = int(f[g].sum()), int(f[e].sum())
    if fp <= 1 and (best is None or tp > best[0] or (tp == best[0] and fp < best[1])):
        best = (tp, fp, t1, t2)
print("best OR rule with <=1 good flagged on TRAIN:", best)
