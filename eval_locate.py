"""
Measure the photo sign-marking feature on REAL palm photos.   (ASCII-only file)
Usage, from the project folder, in the PowerShell window where GEMINI_API_KEY is set:
    python eval_locate.py "C:\\path\\to\\photos"
    python eval_locate.py "C:\\path\\to\\photos" --limit 40 --delay 1.5
    python eval_locate.py "C:\\path\\to\\photos" --verify          (also runs the zoomed second look; needs: pip install pillow)

Folder layout: put photos in sub-folders (or name the files) so that the path contains
    infest ...... palm with confirmed red palm weevil infestation        (e.g. photos\\infested\\a.jpg)
    healthy / clean / control ... palm with no infestation                (e.g. photos\\healthy\\b.jpg)
Only photos whose label comes from a confirmed source should be used (an inspector's report, a dataset label), never a guess.

What it reports, per class: photos analysed, photos with at least one sign marked, average signs per photo, and the sign types.
  - On INFESTED photos, "at least one sign" is how often the model flags visible signs when they should exist.
  - On HEALTHY photos, "at least one sign" is the FALSE-MARK rate: how often it marks signs on a palm without infestation.
With --verify it also reports how many photos keep at least one sign CONFIRMED by the zoomed second look, so you can see whether the second look
lowers the false-mark rate on healthy photos without losing too many infested ones.
It does NOT measure whether a mark is at the right place: that needs photos with hand-drawn reference boxes.
Results are also written to locate_eval.csv (one row per photo).
"""
import os, re, sys, csv, time, asyncio, argparse, mimetypes
from collections import Counter, defaultdict
from locate import locate_signs_from_image

NEG = re.compile(r"(non[-_ ]?infest|un[-_ ]?infest|not[-_ ]?infest|clean|healthy|control)", re.I)
POS = re.compile(r"infest", re.I)
EXT = (".jpg", ".jpeg", ".png", ".webp")

def label_of(rel):
    if NEG.search(rel): return "healthy"
    if POS.search(rel): return "infested"
    return None

async def main():
    ap = argparse.ArgumentParser(); ap.add_argument("folder"); ap.add_argument("--limit", type=int, default=60, help="max photos per class")
    ap.add_argument("--delay", type=float, default=1.0, help="seconds to wait between requests")
    ap.add_argument("--verify", action="store_true", help="also run the zoomed second look"); a = ap.parse_args()
    if not os.environ.get("GEMINI_API_KEY"): print("GEMINI_API_KEY is not set in this window."); return
    by = defaultdict(list); skipped = 0
    for root, _, names in os.walk(a.folder):
        for n in sorted(names):
            if not n.lower().endswith(EXT): continue
            p = os.path.join(root, n); lab = label_of(os.path.relpath(p, a.folder))
            if lab is None: skipped += 1
            else: by[lab].append(p)
    print(f"photos: infested {len(by['infested'])} | healthy {len(by['healthy'])} | unlabelled (ignored) {skipped}")
    if not by["infested"] and not by["healthy"]: print("No labelled photos found. Check the folder names."); return
    rows, stats = [], {}
    for lab in ("infested", "healthy"):
        ok = err = with_sign = 0; n_signs = []; types = Counter(); strong = 0; confirmed = 0
        for p in by[lab][:a.limit]:
            mime = mimetypes.guess_type(p)[0] or "image/jpeg"
            res = await locate_signs_from_image(open(p, "rb").read(), mime, "en", verify=a.verify)
            if res is None: err += 1; rows.append([lab, p, "ERROR", 0, ""]); time.sleep(a.delay); continue
            ok += 1; sg = res["signs"]; n_signs.append(len(sg)); with_sign += bool(sg); confirmed += any(s.get("verified") == "yes" for s in sg); strong += any(s["confidence"] in ("medium", "high") for s in sg)
            types.update(s["type"] for s in sg); rows.append([lab, p, "ok", len(sg), ";".join(f"{s['type']}:{s['confidence']}:{s.get('verified')}" for s in sg)]); time.sleep(a.delay)
        stats[lab] = (ok, err, with_sign, strong, n_signs, types, confirmed)
    print("\n                  analysed  errors  >=1 sign   >=1 medium/high   avg signs" + ("   >=1 CONFIRMED (zoom)" if a.verify else ""))
    for lab in ("infested", "healthy"):
        ok, err, ws, st, ns, ty, cf = stats[lab]
        if ok + err == 0: continue
        print(f"  {lab:<14} {ok:>8} {err:>7} {(100*ws/ok if ok else 0):>7.0f}% {(100*st/ok if ok else 0):>14.0f}% {(sum(ns)/len(ns) if ns else 0):>12.1f}" + (f" {(100*cf/ok if ok else 0):>14.0f}%" if a.verify else "") + f"   types: {dict(ty)}")
    print("\nRead it like this: the rate on INFESTED is how often visible signs are marked; the rate on HEALTHY is the false-mark rate.")
    print("A high false-mark rate means the feature over-marks; a low rate on infested photos means it misses signs. Report both numbers, with the number of photos.")
    with open("locate_eval.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["label", "photo", "status", "signs", "types"]); w.writerows(rows)
    print("saved locate_eval.csv")

if __name__ == "__main__":
    asyncio.run(main())
