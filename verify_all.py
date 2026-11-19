"""
verify_all.py - re-compute every measurable number the project reports, and write a report you can show.   (ASCII-only file)

  python verify_all.py                                                    # offline checks, about 2 minutes
  python verify_all.py --treevibes "G:\\TreeVibes" --esc "G:\\ESC50\\ESC-50-master"     # + real-data checks, about 15 to 25 minutes
  python verify_all.py --poster poster.pptx           # + check that the numbers printed on the poster are in the register

It reads claims.json (the register: every claim, its kind, source and expected range), recomputes what can be recomputed in a TEMPORARY copy
(your deployed model files are never touched), and writes verification_report.json and verification_report.html next to this file.
Claims taken from published papers cannot be recomputed here: the report lists them with their links, to be opened and read.
PASS means: re-computed now, inside the range the register expects. It does NOT mean the number is good, only that it is reproducible and honestly reported.
"""
import os, re, sys, json, math, shutil, hashlib, platform, argparse, subprocess, tempfile, datetime, html, zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def run(args, cwd, timeout=3600):
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    p = subprocess.run([sys.executable] + args, cwd=cwd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout, env=env)
    return (p.stdout or "") + (p.stderr or "")


def workspace():
    tmp = tempfile.mkdtemp(prefix="nk_verify_")
    for f in os.listdir(HERE):
        if f.endswith(".py") and f not in ("app.py", "verify_all.py"):
            shutil.copy(os.path.join(HERE, f), tmp)
    return tmp


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fingerprint(root, marker):
    n = 0; mark = None
    for r, _, names in os.walk(root):
        for x in names:
            if x.lower().endswith(".wav"): n += 1
            if x == marker and mark is None: mark = os.path.join(r, x)
    return {"path": root, "wav_files": n, marker: (sha256(mark) if mark else "not found")}


# ---------------- parsers ----------------
def parse_train_model(out):
    m = {}
    a = re.search(r"accuracy\s+([0-9.]+)\s+(\d+)", out)
    if a: m["sim.accuracy"] = float(a.group(1))
    i = out.find("\u0645\u0635\u0641\u0648\u0641\u0629 \u0627\u0644\u0627\u0644\u062a\u0628\u0627\u0633")          # the Arabic words for "confusion matrix"
    rows = [[int(v) for v in r.split()] for r in re.findall(r"\[\s*(\d[\d\s]*)\]", out[i:])][:4] if i >= 0 else []
    if len(rows) == 4 and all(len(r) == 4 for r in rows):
        inf_total = sum(sum(r) for r in rows[1:]); inf_hit = sum(sum(r[1:]) for r in rows[1:])
        m["sim.detection"] = inf_hit / inf_total
        m["sim.false_alarm"] = sum(rows[0][1:]) / sum(rows[0])
        m["sim.early_recall"] = rows[1][1] / sum(rows[1])
        m["_sim.confusion"] = rows
    return m


def parse_baseline(out):
    m = {}
    a = re.search(r"detection (\d+)/(\d+).*?false alarms (\d+)/(\d+)", out)
    if a: m["baseline.detection"] = int(a.group(1)) / int(a.group(2)); m["baseline.false_alarm"] = int(a.group(3)) / int(a.group(4))
    b = re.search(r"accuracy (\d+)%", out)
    if b: m["baseline.stage_accuracy"] = int(b.group(1)) / 100
    return m


def parse_eval_tv(out):
    a = re.search(r"Detection \(recall\): ([\d.]+)%\s+\|\s+False alarm: ([\d.]+)%", out)
    return {"tv_old.detection": float(a.group(1)) / 100, "tv_old.false_alarm": float(a.group(2)) / 100} if a else {}


def parse_tv_filter(out):
    m = {}
    a = re.search(r"\(chosen on train\):.*?\n\s*detection ([\d.]+)% \| false alarm ([\d.]+)%", out, re.S)
    if a: m["tv_filter.detection"] = float(a.group(1)) / 100; m["tv_filter.false_alarm"] = float(a.group(2)) / 100
    b = re.search(r"ROC AUC on held-out trees: ([\d.]+)", out)
    if b: m["tv_filter.auc"] = float(b.group(1))
    esc = re.findall(r"esc_(\w+)\s+CLEAN\s+\|\s+(\d+)\s+\|\s+([\d.]+)%", out)
    if esc:
        n = sum(int(x[1]) for x in esc); m["esc.flagged_fraction"] = sum(int(x[1]) * float(x[2]) / 100 for x in esc) / n; m["_esc.windows"] = n
    return m


# ---------------- checks ----------------
def check_locator():
    sys.path.insert(0, HERE)
    import numpy as np
    import sensor_location as sl
    def levels(px, py, c=6.5, alpha=0.05):
        d = [math.hypot(px - sx, py - sy) + 1 for sx, sy in sl.SENSORS]
        return [math.exp(c - alpha * di - sl.BETA * math.log(di)) for di in d]
    worst = 0.0
    for px in (8, 14, 20, 26, 32):
        for py in (12, 24, 36, 48, 60):
            r = sl.locate_from_rms(levels(px, py)); worst = max(worst, 99.0 if r is None else math.hypot(r["x"] - px, r["y"] - py))
    rng = np.random.default_rng(0); errs = []; lost = 0
    for _ in range(300):
        px, py = rng.uniform(8, 32), rng.uniform(12, 68); lv = [v * (1 + rng.normal(0, 0.05)) for v in levels(px, py)]; r = sl.locate_from_rms(lv)
        if r is None: lost += 1
        else: errs.append(math.hypot(r["x"] - px, r["y"] - py))
    return {"locator.max_error_cm_noise_free": worst, "locator.median_error_cm_5pct_noise": float(np.median(errs)) if errs else None, "_locator.lost_of_300": lost}


def check_poster(path, claims):
    z = zipfile.ZipFile(path); text = " ".join(re.sub(r"<[^>]+>", " ", z.read(n).decode("utf-8", "ignore")) for n in z.namelist() if n.startswith("ppt/slides/slide"))
    out = []
    for c in claims:
        for s in c.get("poster", []):
            out.append({"claim": c["id"], "text": s, "found": s in text})
    return out


def judge(c, metrics):
    if "metric" not in c: return ("info", None)
    v = metrics.get(c["metric"])
    if v is None: return ("skipped", None)
    e = c.get("expect")
    if not e: return ("reported", v)
    return ("pass" if e[0] <= v <= e[1] else "fail", v)


def fmt(v, metric):
    if v is None: return "-"
    if metric.endswith(("accuracy", "detection", "false_alarm", "early_recall", "fraction")): return f"{v*100:.1f}%"
    return f"{v:.2f}"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--treevibes", default=""); ap.add_argument("--esc", default=""); ap.add_argument("--poster", default=""); ap.add_argument("--out", default=HERE)
    a = ap.parse_args()
    reg = json.load(open(os.path.join(HERE, "claims.json"), encoding="utf-8")); metrics = {}; log = []; fps = {}
    print("1/5 simulation: training the 4-stage model in a temporary copy ...")
    tmp = workspace()
    try:
        out = run(["train_model.py"], tmp); metrics.update(parse_train_model(out)); log.append("train_model.py")
        print("2/5 baseline check ...")
        out = run(["baseline_check.py"], tmp); metrics.update(parse_baseline(out)); log.append("baseline_check.py")
        print("3/5 location from three sensors (simulation) ...")
        metrics.update(check_locator()); log.append("sensor_location.py")
        if a.treevibes:
            fps["treevibes"] = fingerprint(a.treevibes, "TreeVibes.xlsx")
            print("4/5 the old model on real TreeVibes recordings ...")
            metrics.update(parse_eval_tv(run(["eval_treevibes.py", a.treevibes], tmp))); log.append("eval_treevibes.py")
            if a.esc:
                fps["esc50"] = fingerprint(a.esc, os.path.join("esc50.csv"))
                print("5/5 noise filter: trees-only test, then held-out wind and insects (two trainings, several minutes each) ...")
                base = ["train_treevibes.py", "train", a.treevibes, "--noise-dir", a.esc, "--add-simulated", "60"]
                o1 = run(base + ["--noise-holdout", "none"], tmp); m1 = parse_tv_filter(o1)
                metrics.update({k: v for k, v in m1.items() if k.startswith("tv_filter")})
                o2 = run(base, tmp); m2 = parse_tv_filter(o2)
                metrics.update({k: v for k, v in m2.items() if k.startswith("esc") or k.startswith("_esc")}); log += ["train_treevibes.py (x2)"]
            else:
                print("5/5 skipped: give --esc to check the noise filter")
        else:
            print("4/5 and 5/5 skipped: give --treevibes (and --esc) to check the real-data claims")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    rows = []; counts = {"pass": 0, "fail": 0, "reported": 0, "skipped": 0}
    for c in reg["claims"]:
        if c["kind"] in ("measured_sim", "measured_real"):
            st, v = judge(c, metrics); counts[st] += 1
            rows.append({"id": c["id"], "kind": c["kind"], "ar": c["ar"], "en": c["en"], "shown": c.get("shown"), "metric": c.get("metric"), "expect": c.get("expect"), "measured": v, "measured_text": fmt(v, c.get("metric", "")), "status": st, "reproduce": c.get("reproduce")})
    poster = check_poster(a.poster, reg["claims"]) if a.poster else None
    rep = {"time": datetime.datetime.now().isoformat(timespec="seconds"), "environment": {"python": platform.python_version(), "os": platform.platform(), "numpy": __import__("numpy").__version__, "scikit_learn": __import__("sklearn").__version__}, "datasets": fps, "ran": log, "counts": counts, "checks": rows, "poster": poster, "extra_metrics": {k: v for k, v in metrics.items() if k.startswith("_")}}
    jp = os.path.join(a.out, "verification_report.json"); json.dump(rep, open(jp, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    badge = {"pass": ("#2e7d4f", "PASS"), "fail": ("#b3342d", "FAIL"), "reported": ("#1F618D", "REPORTED"), "skipped": ("#8a8f8c", "SKIPPED")}
    tr = "".join(f"<tr><td>{r['id']}</td><td>{html.escape(r['ar'])}<br><span class='en'>{html.escape(r['en'])}</span></td><td>{html.escape(str(r['shown']))}</td><td><b>{html.escape(r['measured_text'])}</b></td><td><span class='b' style='background:{badge[r['status']][0]}'>{badge[r['status']][1]}</span></td><td><code>{html.escape(r['reproduce'] or '')}</code></td></tr>" for r in rows)
    ptxt = "" if poster is None else "<h2>Poster numbers</h2><table><tr><th>claim</th><th>text on the poster</th><th>found</th></tr>" + "".join(f"<tr><td>{p['claim']}</td><td>{html.escape(p['text'])}</td><td>{'yes' if p['found'] else 'NO'}</td></tr>" for p in poster) + "</table>"
    fp = "".join(f"<li>{k}: {html.escape(json.dumps(v, ensure_ascii=False))}</li>" for k, v in fps.items()) or "<li>no dataset given</li>"
    page = f"""<!doctype html><html lang="ar" dir="rtl"><meta charset="utf-8"><title>NakhlaGuard verification report</title><style>body{{font-family:Tahoma,Arial,sans-serif;max-width:1000px;margin:24px auto;padding:0 16px;color:#14201B}}table{{border-collapse:collapse;width:100%;font-size:.9rem}}td,th{{border:1px solid #ddd;padding:7px;vertical-align:top;text-align:start}}th{{background:#1F5C4A;color:#fff}}.b{{color:#fff;padding:2px 9px;border-radius:9px;font-size:.78rem}}.en{{color:#555;font-size:.82rem}}code{{font-size:.78rem}}</style>
<h1>NakhlaGuard: verification report / &#1578;&#1602;&#1585;&#1610;&#1585;&#32;&#1575;&#1604;&#1578;&#1581;&#1602;&#1602;</h1><p>{rep['time']} | Python {rep['environment']['python']} | scikit-learn {rep['environment']['scikit_learn']} | numpy {rep['environment']['numpy']} | {html.escape(rep['environment']['os'])}</p>
<p><b>PASS {counts['pass']} | FAIL {counts['fail']} | REPORTED {counts['reported']} | SKIPPED {counts['skipped']}</b>. PASS means: re-computed now and inside the range the register expects; it does not mean the number is good. Claims from published papers are not recomputed here: see evidence.html for their links.</p>
<h2>Datasets (fingerprints)</h2><ul>{fp}</ul><table><tr><th>#</th><th>Claim</th><th>Shown</th><th>Measured now</th><th>Status</th><th>Reproduce</th></tr>{tr}</table>{ptxt}</html>"""
    open(os.path.join(a.out, "verification_report.html"), "w", encoding="utf-8").write(page)
    print(f"\nPASS {counts['pass']} | FAIL {counts['fail']} | REPORTED {counts['reported']} | SKIPPED {counts['skipped']}")
    for r in rows: print(f"  {r['id']:<3} {r['status'].upper():<8} shown {str(r['shown']):<9} measured {r['measured_text']}")
    print("\nwrote verification_report.html and verification_report.json")


if __name__ == "__main__":
    main()
