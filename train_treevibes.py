"""
Train and EVALUATE a real-data detector (healthy vs infested) on the public TreeVibes recordings.   (ASCII-only file)

Modes (run from the project folder):
  train    python train_treevibes.py train "G:\\TreeVibes" --noise-dir "G:\\ESC50\\ESC-50-master" --add-simulated 60 --noise-holdout none
           Trains the model and saves nakhla_binary_model.joblib (the noise filter the website loads). Reports results on held-out trees.
  cv       python train_treevibes.py cv "G:\\TreeVibes" --noise-dir "G:\\ESC50\\ESC-50-master" --add-simulated 60
           Leave-tree-out cross-validation: every tree and every noise category is tested while the model never saw it.
           This number is far steadier than the single train/test split, so USE IT to compare ideas. It does not save a model.
  predict  python train_treevibes.py predict some_recording.wav

Things to try (compare the "cv" numbers, then train the winner for the website):
  --features extra         adds 14 features about impulsive, repeating pulses (features_extra.py)
  --windows-per-file 3     several 5 s windows from each file, not only the middle one (use --max-per-tree 300)
  --use-lab 600            adds real lab recordings (clean vs infested, labelled per file) to training
  --model hgb | et | rf    gradient boosting, extra trees or random forest (default rf)
  --sr 16000               convert recordings to 16 kHz instead of 4 kHz: tells you whether the 2 kHz limit of the node costs accuracy (analysis only)
  --own-noise-dir "G:\\my_noise"   noise recorded with YOUR sensor, one sub-folder per type (wind, car, birds...). Usually the most valuable data you can add.

Rules that keep the numbers honest: the split is by tree, never by clip; the threshold is chosen on training trees only;
the held-out trees are never used to tune anything; choose a setting by cv, then report the cv numbers (not the best of many runs of a single split).
"""
import os, re, sys, argparse, warnings, random
from math import gcd
from collections import defaultdict, Counter
import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly
warnings.filterwarnings("ignore")
import joblib
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
from features import extract_features, FEATURE_NAMES

TARGET_SR, WIN_S = 4000, 5.0
CLEAN_N = {7, 8, 9, 10, 24, 25, 35}
LABEL_N = {n: (0 if n in CLEAN_N else 1) for n in range(1, 36)}
NEG = re.compile(r"(non[-_ ]?infest|un[-_ ]?infest|not[-_ ]?infest|clean|healthy|control)", re.I)
POS = re.compile(r"infest", re.I)
FOLDER_RE = re.compile(r"folder[_ -]?(\d+)", re.I)
MODEL_OUT = "nakhla_binary_model.joblib"
FEAT = {"key": "base", "fn": extract_features, "names": FEATURE_NAMES}
# ESC-50 (environmental sounds, github.com/karolpiczak/ESC-50): categories that can appear around a palm and cause false alarms
ESC_DEFAULT = "wind,rain,thunderstorm,insects,crickets,chirping_birds,rooster,crow,cow,sheep,engine,chainsaw,hand_saw,airplane,helicopter,train,car_horn,siren,footsteps,door_wood_knock,clock_tick,vacuum_cleaner"

def set_features(key):
    if key == "extra":
        from features_extra import extra_features, FEATURE_NAMES_EXTRA
        FEAT.update({"key": "extra", "fn": extra_features, "names": FEATURE_NAMES_EXTRA})
    else:
        FEAT.update({"key": "base", "fn": extract_features, "names": FEATURE_NAMES})

def load_esc(root, cats):
    """ESC-50 folder: audio/*.wav and meta/esc50.csv. Returns {category: [wav paths]} or None if the layout is not found."""
    import csv
    meta, audio = os.path.join(root, "meta", "esc50.csv"), os.path.join(root, "audio")
    if not (os.path.exists(meta) and os.path.isdir(audio)): return None
    by = defaultdict(list)
    with open(meta, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if cats is None or r["category"] in cats: by[r["category"]].append(os.path.join(audio, r["filename"]))
    return by

def label_of(rel):
    low = rel.lower()
    if NEG.search(low): return 0
    if POS.search(low): return 1
    m = FOLDER_RE.findall(rel)
    if m and int(m[-1]) in LABEL_N: return LABEL_N[int(m[-1])]
    return None

def tree_of(rel, lab):
    m = FOLDER_RE.findall(rel)
    return int(m[-1]) if m else ("lab_clean" if lab == 0 else "lab_infested")

def load_wav(path):
    sr, a = wavfile.read(path)
    if a.ndim > 1: a = a.mean(axis=1)
    a = a.astype(np.float64); a -= a.mean(); a /= (np.max(np.abs(a)) + 1e-9)
    if sr != TARGET_SR:
        g = gcd(int(sr), TARGET_SR); a = resample_poly(a, TARGET_SR // g, int(sr) // g)
    return a.astype(np.float32)

def windows_of(x, n):
    """Up to n evenly spaced 5-second windows (short clips are repeated up to 5 s)."""
    w = int(WIN_S * TARGET_SR)
    if len(x) < TARGET_SR: return []
    if len(x) < w: x = np.resize(x, w)
    k = max(1, min(n, len(x) // w))
    starts = [int(s) for s in np.linspace(0, len(x) - w, k)] if k > 1 else [(len(x) - w) // 2]
    return [x[s:s + w] for s in starts]

def feats_of(path, nwin=1):
    try: x = load_wav(path)
    except Exception: return []
    return [FEAT["fn"](win, TARGET_SR) for win in windows_of(x, nwin)]

def make_model(kind, n_est, seed):
    if kind == "et": return ExtraTreesClassifier(n_estimators=n_est, max_depth=14, random_state=seed, n_jobs=-1)
    if kind == "hgb": return HistGradientBoostingClassifier(max_depth=6, learning_rate=0.06, max_iter=max(100, n_est), random_state=seed)
    return RandomForestClassifier(n_estimators=n_est, max_depth=12, random_state=seed, n_jobs=-1)

def weights(yy, gg):                      # every tree equal, both classes balanced
    cnt = Counter(gg); w = np.array([1.0 / cnt[t] for t in gg])
    for c in (0, 1): w[yy == c] /= w[yy == c].sum()
    return w * len(w) / 2

def collect(a):
    random.seed(1); np.random.seed(1)
    test_clean = {int(v) for v in a.test_clean.split(",") if v.strip()}
    by_tree = defaultdict(list); unl = 0
    for root, _, names in os.walk(a.folder):
        for n in names:
            if not n.lower().endswith(".wav"): continue
            p = os.path.join(root, n); rel = os.path.relpath(p, a.folder); lab = label_of(rel)
            if lab is None: unl += 1; continue
            parts = [s.lower() for s in rel.split(os.sep)]
            t = tree_of(rel, lab)
            is_lab = isinstance(t, str)
            is_test = (isinstance(t, int) and t in test_clean) or ("test" in parts)
            by_tree[(t, lab, "test" if is_test else ("lab" if is_lab else "train"))].append(p)
    print(f"trees/groups found: {len(by_tree)} | unlabelled files ignored: {unl} | features: {FEAT['key']} ({len(FEAT['names'])}) | windows per file: {a.windows_per_file}")
    rows = []   # (features, label, tree, split)
    for (t, lab, split), fl in sorted(by_tree.items(), key=lambda kv: str(kv[0])):
        cap = a.use_lab if split == "lab" else a.max_per_tree
        if split == "lab" and a.use_lab == 0: continue
        random.shuffle(fl); got = 0
        for p in fl:
            for f in feats_of(p, a.windows_per_file):
                rows.append((f, lab, str(t), "train" if split == "lab" else split)); got += 1
                if got >= cap: break
            if got >= cap: break
        print(f"  {str(t):<12} {'infested' if lab else 'clean   '} {split:<5} windows used: {got}")
    if a.noise_dir:
        cats = None if a.noise_categories == "all" else {c.strip() for c in a.noise_categories.split(",") if c.strip()}
        hold = {c.strip() for c in a.noise_holdout.split(",") if c.strip()}
        esc = load_esc(a.noise_dir, cats)
        if not esc: print(f"ESC-50 layout not found in {a.noise_dir} (expected audio/ and meta/esc50.csv). Skipping noise data.")
        else:
            print("\nNoise / other-sound data (ESC-50), label = not infested:")
            for cat in sorted(esc):
                fl = esc[cat][:]; random.shuffle(fl); got = 0
                for p in fl:
                    for f in feats_of(p, 1):
                        rows.append((f, 0, "esc_" + cat, "test" if cat in hold else "train")); got += 1
                    if got >= a.noise_per_cat: break
                print(f"  esc_{cat:<18} {'HELD OUT (never trained on)' if cat in hold else 'train':<27} windows used: {got}")
    if a.own_noise_dir:
        own = defaultdict(list)
        for root, _, names in os.walk(a.own_noise_dir):
            for n in names:
                if n.lower().endswith(".wav"):
                    sub = os.path.relpath(root, a.own_noise_dir).split(os.sep)[0]
                    own["own_" + ("root" if sub == "." else sub)].append(os.path.join(root, n))
        print("\nYour OWN noise recordings (made with your sensor, label = not infested; each sub-folder is one noise type):")
        for cat, fl in sorted(own.items()):
            random.shuffle(fl); got = 0
            for p in fl:
                for f in feats_of(p, a.windows_per_file):
                    rows.append((f, 0, "esc_" + cat, "train")); got += 1
                    if got >= a.noise_per_cat * 3: break
                if got >= a.noise_per_cat * 3: break
            print(f"  {cat:<24} train windows used: {got}")
    if a.add_simulated:
        try:
            from generate_synthetic_data import generate_sample, CLASSES as SIM_CLASSES, SAMPLE_RATE as SIM_SR
        except Exception as e:
            print("simulated data module not found, skipping:", type(e).__name__)
        else:
            rng = np.random.default_rng(123)
            for ci, cls in enumerate(SIM_CLASSES):
                for _ in range(a.add_simulated):
                    rows.append((FEAT["fn"](generate_sample(cls, rng), SIM_SR), 0 if ci == 0 else 1, "sim_%d" % ci, "train"))
            print(f"\nsimulated clips added to TRAINING only: {a.add_simulated} per stage (stage 'healthy' = not infested, the other three = infested)")
    return rows

def cmd_train(a):
    rows = collect(a)
    X = np.array([r[0] for r in rows]); y = np.array([r[1] for r in rows]); g = np.array([r[2] for r in rows]); sp = np.array([r[3] for r in rows])
    tr, te = sp == "train", sp == "test"
    print(f"\ntrain windows: {tr.sum()} (clean {(y[tr]==0).sum()} | infested {(y[tr]==1).sum()}, trees {len(set(g[tr]))})"
          f" | test windows: {te.sum()} (clean {(y[te]==0).sum()} | infested {(y[te]==1).sum()}, trees {len(set(g[te]))})")
    if te.sum() == 0 or len(set(y[tr])) < 2 or len(set(y[te])) < 2:
        print("Not enough data in train or test for both classes. Check the folder path / --test-clean."); return
    # 1) threshold from out-of-fold predictions on the TRAIN trees
    oof = np.zeros(tr.sum()); Xt, yt, gt = X[tr], y[tr], g[tr]
    for fi, (i_tr, i_va) in enumerate(GroupKFold(n_splits=min(5, len(set(gt)))).split(Xt, yt, gt)):
        m = make_model(a.model, 150, fi)
        m.fit(Xt[i_tr], yt[i_tr], sample_weight=weights(yt[i_tr], gt[i_tr])); oof[i_va] = m.predict_proba(Xt[i_va])[:, 1]
    cands = []
    for th in np.arange(0.05, 0.96, 0.01):
        far = np.mean(oof[yt == 0] >= th); rec = np.mean(oof[yt == 1] >= th)
        if far <= a.target_false_alarm: cands.append((float(th), rec, far))
    if cands:
        top = max(c[1] for c in cands)                                  # best detection reachable within the false-alarm budget
        best = max((c for c in cands if c[1] >= top - 0.005), key=lambda c: c[0])   # the HIGHEST threshold that keeps it (fewest false alarms)
    else:
        best = (0.5, np.mean(oof[yt == 1] >= .5), np.mean(oof[yt == 0] >= .5)); print("target false-alarm not reachable; using 0.5")
    th = float(best[0])
    print(f"\nthreshold chosen on training trees (out-of-fold): {th:.2f}  -> recall {best[1]*100:.0f}% | false alarm {best[2]*100:.0f}%")
    # 2) final model on all train trees, evaluated ONCE on held-out test trees
    clf = make_model(a.model, 300, 42)
    clf.fit(Xt, yt, sample_weight=weights(yt, gt))
    p = clf.predict_proba(X[te])[:, 1]; yte, gte = y[te], g[te]
    print("\nHELD-OUT TREES (never seen in training):  tree | truth | windows | flagged as infested")
    for t in sorted(set(gte), key=str):
        mk = gte == t; print(f"  {t:<10} {'infested' if yte[mk][0] else 'CLEAN   '} | {mk.sum():>4} | {100*np.mean(p[mk] >= th):5.1f}%")
    for name, thr in (("threshold %.2f (chosen on train)" % th, th), ("default 0.50", 0.5)):
        pr = p >= thr; tn = int(((yte == 0) & ~pr).sum()); fp = int(((yte == 0) & pr).sum()); fn = int(((yte == 1) & ~pr).sum()); tp = int(((yte == 1) & pr).sum())
        rec = tp / max(tp + fn, 1); far = fp / max(fp + tn, 1)
        print(f"\n{name}:   clean -> healthy {tn}, false alarm {fp}   |   infested -> missed {fn}, detected {tp}")
        print(f"   detection {rec*100:.1f}% | false alarm {far*100:.1f}% | balanced accuracy {(rec+(1-far))/2*100:.1f}%")
    print(f"\nROC AUC on held-out trees: {roc_auc_score(yte, p):.3f}   (0.5 = coin flip, 1.0 = perfect)")
    if hasattr(clf, "feature_importances_"):
        imp = sorted(zip(FEAT["names"], clf.feature_importances_), key=lambda z: -z[1])[:6]
        print("top features:", ", ".join(f"{n} {v:.3f}" for n, v in imp))
    joblib.dump({"model": clf, "threshold": th, "feature_names": FEAT["names"], "features": FEAT["key"], "model_kind": a.model, "sr": TARGET_SR, "note": "trained on TreeVibes, split by tree"}, MODEL_OUT)
    print(f"\nsaved {MODEL_OUT}")
    print("Read this honestly: clean trees are few and mostly NOT palms, the sensor differs from ours, and recordings come from Crete.")

def cmd_cv(a):
    rows = collect(a)
    X = np.array([r[0] for r in rows]); y = np.array([r[1] for r in rows]); g = np.array([r[2] for r in rows])
    fixed = np.array([x.startswith("sim_") or x.startswith("lab_") for x in g])           # always in training, never evaluated
    ev = np.where(~fixed)[0]; fx = np.where(fixed)[0]
    if len(set(y[ev])) < 2: print("Need both clean and infested trees to evaluate."); return
    k = min(5, len(set(g[ev])))
    oof = np.full(len(y), np.nan); fold_auc = []
    for fi, (i_tr, i_te) in enumerate(GroupKFold(n_splits=k).split(ev, y[ev], g[ev])):
        tr_idx = np.concatenate([ev[i_tr], fx]); te_idx = ev[i_te]
        m = make_model(a.model, 200, fi); m.fit(X[tr_idx], y[tr_idx], sample_weight=weights(y[tr_idx], g[tr_idx]))
        oof[te_idx] = m.predict_proba(X[te_idx])[:, 1]
        tt = te_idx[[not gg.startswith("esc_") for gg in g[te_idx]]]
        if len(tt) and len(set(y[tt])) == 2: fold_auc.append(roc_auc_score(y[tt], oof[tt]))
        print(f"  fold {fi + 1}/{k} done")
    is_tree = np.array([not gg.startswith("esc_") for gg in g[ev]]); tree_i, esc_i = ev[is_tree], ev[~is_tree]
    auc = roc_auc_score(y[tree_i], oof[tree_i])
    thr = float(np.quantile(oof[tree_i[y[tree_i] == 0]], 0.8))                              # 20% of the CLEAN tree windows would be flagged
    det = float(np.mean(oof[tree_i[y[tree_i] == 1]] >= thr)); noise = float(np.mean(oof[esc_i] >= thr)) if len(esc_i) else None
    print(f"\nLEAVE-TREE-OUT CROSS-VALIDATION ({k} folds; every tree and noise category is tested without the model having seen it)")
    print(f"  setting: features={FEAT['key']} model={a.model} windows/file={a.windows_per_file} max/tree={a.max_per_tree} lab={a.use_lab} simulated={a.add_simulated} noise={'yes' if a.noise_dir else 'no'}")
    print(f"  ROC AUC on real trees (pooled): {auc:.3f}" + (f"   | per fold: mean {np.mean(fold_auc):.3f}, std {np.std(fold_auc):.3f}, values {[round(v, 2) for v in fold_auc]}" if fold_auc else ""))
    print(f"  at the point where 20% of CLEAN tree windows are flagged: detection {det*100:.0f}%" + (f" | environmental noise flagged {noise*100:.0f}%" if noise is not None else ""))
    print("  per tree (share of windows scoring above that point):")
    for t_ in sorted({x for x in g[tree_i]}, key=lambda z: int(z)):
        mk = tree_i[g[tree_i] == t_]; print(f"    tree {t_:<4} {'infested' if y[mk][0] else 'CLEAN   '} {100*np.mean(oof[mk] >= thr):5.0f}%   (n={len(mk)})")
    if len(esc_i):
        print("  noise categories flagged:", {c[4:]: round(float(np.mean(oof[esc_i[g[esc_i] == c]] >= thr)), 2) for c in sorted({x for x in g[esc_i]})})
    print("\nCompare settings by AUC (higher is better, 0.5 is a coin flip). Differences under about 0.03 are within the noise of 30-ish trees.")
    print("To use a setting on the website: run the same options with mode 'train' (and --noise-holdout none).")

def cmd_predict(a):
    d = joblib.load(MODEL_OUT); set_features(d.get("features", "base")); x = load_wav(a.folder); w = int(WIN_S * TARGET_SR)
    if len(x) < w: x = np.resize(x, w)
    ps = [d["model"].predict_proba(FEAT["fn"](x[i:i + w], TARGET_SR).reshape(1, -1))[0, 1] for i in range(0, len(x) - w + 1, w)][:12]
    p = float(np.mean(ps)); print(f"probability infested: {p*100:.0f}%  ->  {'INFESTED' if p >= d['threshold'] else 'healthy'}  (threshold {d['threshold']:.2f}, {len(ps)} windows)")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("mode", choices=["train", "cv", "predict"]); ap.add_argument("folder", help="dataset folder (train, cv) or a wav file (predict)")
    ap.add_argument("--max-per-tree", type=int, default=150); ap.add_argument("--target-false-alarm", type=float, default=0.2)
    ap.add_argument("--test-clean", default="9,35", help="clean tree numbers held out for testing (train mode)"); ap.add_argument("--use-lab", type=int, default=0, help="also use N lab windows per class for training (0 = no)")
    ap.add_argument("--noise-dir", default="", help="ESC-50 folder: adds wind, rain, insects, birds, engines... as NOT-infested examples")
    ap.add_argument("--noise-categories", default=ESC_DEFAULT, help="comma list of ESC-50 categories, or 'all'")
    ap.add_argument("--noise-holdout", default="wind,insects", help="noise categories kept out of training to test unseen noise (train mode; use 'none' for the final model)")
    ap.add_argument("--noise-per-cat", type=int, default=40, help="max clips per noise category")
    ap.add_argument("--add-simulated", type=int, default=0, help="add N simulated clips per stage to training (keeps the simulated demo recordings working with the noise filter)")
    ap.add_argument("--own-noise-dir", default="", help="your own noise recordings: one sub-folder per noise type (wind, car, birds...), all treated as NOT infested")
    ap.add_argument("--features", choices=["base", "extra"], default="base", help="extra = +14 features about impulsive, repeating pulses")
    ap.add_argument("--windows-per-file", type=int, default=1, help="5 s windows taken from each file")
    ap.add_argument("--model", choices=["rf", "et", "hgb"], default="rf", help="random forest, extra trees or gradient boosting")
    ap.add_argument("--sr", type=int, default=4000, help="sample rate recordings are converted to. 4000 = what the Arduino node records. Higher values test whether the 2 kHz limit hurts (analysis only, see notes)")
    a = ap.parse_args()
    TARGET_SR = a.sr
    if a.sr != 4000:
        print(f"NOTE: sample rate {a.sr} Hz. This is an ANALYSIS experiment: the website and the Arduino node work at 4000 Hz. Simulated clips are not added at other rates, and the model is saved under another name.")
        MODEL_OUT = f"nakhla_binary_model_sr{a.sr}.joblib"; a.add_simulated = 0
    if a.mode == "predict": cmd_predict(a)
    else:
        set_features(a.features); (cmd_train if a.mode == "train" else cmd_cv)(a)
