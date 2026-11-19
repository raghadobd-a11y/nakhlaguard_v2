"""
Evaluate the NakhlaGuard model on the public TreeVibes recordings.  (ASCII-only file: safe to save in Notepad)
Usage, from the project folder (after python train_model.py has been run once):
    python eval_treevibes.py "G:\\TreeVibes"
    python eval_treevibes.py "G:\\TreeVibes\\lab"            (only the lab part)
    python eval_treevibes.py "G:\\TreeVibes" --per-folder 300 --windows 2

How labels are found, for every wav file (path relative to the folder you give):
  1) a folder or file name containing "infest" = infested ; "clean" / "healthy" / "control" / "non-infested" = clean
  2) otherwise a folder named folder_N, using the AUDIO column of TreeVibes.xlsx (1 = pest sounds, 0 = none)
Files it cannot label are ignored, and a few example paths are printed so the rule can be adapted.
Results are grouped by folder (a group = the last 3 parts of the file's folder path).
"""
import os, re, argparse, warnings, random
from math import gcd
from collections import Counter, defaultdict
import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly
warnings.filterwarnings("ignore")
from predict import predict_severity_from_array

TARGET_SR, WIN_S = 4000, 5.0
HEALTHY = "\u0633\u0644\u064a\u0645"                       # the model's name for the healthy class
CLEAN = {7, 8, 9, 10, 24, 25, 35}                            # AUDIO = 0 in TreeVibes.xlsx
LABEL_N = {n: (0 if n in CLEAN else 1) for n in range(1, 36)}
NEG = re.compile(r"(non[-_ ]?infest|un[-_ ]?infest|not[-_ ]?infest|clean|healthy|control)", re.I)
POS = re.compile(r"infest", re.I)
FOLDER_RE = re.compile(r"folder[_ -]?(\d+)", re.I)

def label_of(rel):
    low = rel.lower()
    if NEG.search(low): return 0
    if POS.search(low): return 1
    m = FOLDER_RE.findall(rel)
    if m and int(m[-1]) in LABEL_N: return LABEL_N[int(m[-1])]
    return None

def load_wav(path):
    sr, a = wavfile.read(path)
    if a.ndim > 1: a = a.mean(axis=1)
    a = a.astype(np.float64); a -= a.mean(); a /= (np.max(np.abs(a)) + 1e-9)
    if sr != TARGET_SR:
        g = gcd(int(sr), TARGET_SR); a = resample_poly(a, TARGET_SR // g, int(sr) // g)
    return a.astype(np.float32)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("folder")
    ap.add_argument("--per-folder", type=int, default=300, help="max files sampled from each group")
    ap.add_argument("--windows", type=int, default=2, help="max 5-second windows per file")
    a = ap.parse_args()
    groups = defaultdict(list); unl = []; n_all = 0
    for root, _, names in os.walk(a.folder):
        for n in names:
            if not n.lower().endswith(".wav"): continue
            n_all += 1; p = os.path.join(root, n); rel = os.path.relpath(p, a.folder)
            lab = label_of(rel)
            if lab is None:
                if len(unl) < 6: unl.append(rel)
                continue
            parts = rel.split(os.sep)[:-1]
            groups[("/".join(parts[-3:]) or "(root)", lab)].append(p)
    nlab = sum(len(v) for v in groups.values())
    print(f"wav files found: {n_all} | labelled: {nlab} | unlabelled (ignored): {n_all - nlab}")
    if unl: print("examples of UNLABELLED files:", *["\n   " + u for u in unl])
    if not groups:
        print("Nothing could be labelled. Send me this whole output."); return
    print("groups:", {f"{g}={'infested' if l else 'clean'}": len(v) for (g, l), v in groups.items()})
    random.seed(1); w = int(WIN_S * TARGET_SR)
    rows = []; bad = padded = tiny = 0
    for (g, lab), fl in sorted(groups.items()):
        fl = fl[:]; random.shuffle(fl)
        for p in fl[:a.per_folder]:
            try: x = load_wav(p)
            except Exception: bad += 1; continue
            if len(x) < TARGET_SR: tiny += 1; continue
            if len(x) < w: x = np.resize(x, w); padded += 1
            nwin = len(x) // w; idx = np.linspace(0, nwin - 1, min(nwin, a.windows)).astype(int)
            for i in idx:
                r = predict_severity_from_array(x[i * w:(i + 1) * w], TARGET_SR)
                rows.append((g, lab, r["predicted_class"]))
    if bad or padded or tiny: print(f"unreadable: {bad} | clips shorter than 5 s repeated to 5 s: {padded} | clips shorter than 1 s skipped: {tiny}")
    if not rows: print("No usable windows."); return
    L = np.array([r[1] for r in rows]); P = np.array([r[2] != HEALTHY for r in rows])
    print("\nPer group  (truth | windows | flagged as infested):")
    for (g, lab) in sorted({(r[0], r[1]) for r in rows}):
        ws = [r for r in rows if r[0] == g and r[1] == lab]
        print(f"  {g:<34} {'infested' if lab else 'CLEAN   '} | {len(ws):>5} | {100*np.mean([r[2] != HEALTHY for r in ws]):5.1f}%")
    TN = int(((L == 0) & ~P).sum()); FP = int(((L == 0) & P).sum()); FN = int(((L == 1) & ~P).sum()); TP = int(((L == 1) & P).sum())
    print(f"\n5-second windows: {len(rows)}   (clean {(L==0).sum()} | infested {(L==1).sum()})")
    print("                  pred healthy   pred infested")
    print(f"true clean        {TN:>10}   {FP:>12}")
    print(f"true infested     {FN:>10}   {TP:>12}")
    rec = TP / max(TP + FN, 1); far = FP / max(FP + TN, 1)
    print(f"\nDetection (recall): {rec*100:.1f}%  |  False alarm: {far*100:.1f}%  |  Balanced accuracy: {(rec+(1-far))/2*100:.1f}%")
    print("Stages predicted for infested windows:", dict(Counter(r[2] for r in rows if r[1] == 1)))
    print("\nRead this honestly: the current model was trained on SIMULATED data. Real recordings use a different sensor and conditions.")

if __name__ == "__main__":
    main()
