"""
Baseline check for the poster: how much better is the Random Forest than a trivial one-feature threshold?   (ASCII-only)
Run from the project folder, after  python train_model.py  :     python baseline_check.py
Uses the SAME simulated data and the SAME 75/25 split as train_model.py. The baseline uses ONE feature
(band_energy_ratio_100_800hz); its thresholds are fitted on the training part only.
"""
import json, os
import numpy as np
from sklearn.model_selection import train_test_split
from features import extract_features, FEATURE_NAMES

RAW = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_raw")
man = json.load(open(os.path.join(RAW, "manifest.json"), encoding="utf-8"))
X = np.array([extract_features(np.load(os.path.join(RAW, it["file"]))) for it in man]); y = np.array([it["label_id"] for it in man])
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
fi = FEATURE_NAMES.index("band_energy_ratio_100_800hz"); f_tr, f_te = Xtr[:, fi], Xte[:, fi]
th = max(((np.mean((f_tr >= t) == (ytr > 0)), t) for t in np.unique(f_tr)), key=lambda z: z[0])[1]
inf = f_te >= th
tp = int(((yte > 0) & inf).sum()); fn = int(((yte > 0) & ~inf).sum()); fp = int(((yte == 0) & inf).sum()); tn = int(((yte == 0) & ~inf).sum())
print(f"healthy vs infested, one-feature threshold: detection {tp}/{tp+fn} = {100*tp/(tp+fn):.0f}% | false alarms {fp}/{fp+tn} = {100*fp/(fp+tn):.0f}%")
means = np.array([f_tr[ytr == c].mean() for c in range(4)]); order = np.argsort(means); cuts = [(means[order[i]] + means[order[i + 1]]) / 2 for i in range(3)]
pred4 = np.array([order[np.searchsorted(cuts, v)] for v in f_te])
print(f"4-stage classification, same feature with 3 cuts: accuracy {100*np.mean(pred4 == yte):.0f}% ({int((pred4 == yte).sum())}/{len(yte)})")
print("Compare with the Random Forest accuracy printed by train_model.py (93% on the poster).")
