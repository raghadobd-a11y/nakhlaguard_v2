"""
False-alarm test for the "Record now" / upload path of the public try page.   Run:  python train_model.py   (once)   then   python verify_noise.py

It makes many SYNTHETIC non-weevil sounds (air, wind on the microphone, breath, fan, hum, speech-like, rustle, traffic ...), turns each one
into a 4 kHz WAV exactly like the web page does, and sends it through the SAME code the site uses (demo_audio.analyze_clip with the trained
model, the fusion step and the pulse check). It counts how often the site would show an infestation stage for a sound that has no larvae in it,
and how often it still hears the simulated weevil sounds.

What this proves and what it does not:
  * it proves how the rules behave on these made-up sounds, and lets us tighten them without breaking the simulated weevil;
  * it does NOT prove anything about real microphones or real palms. Real recordings are the next step (README, "What is not tested yet").
  * short sharp taps (fingernail, keyboard) really are short pulses; the pulse check cannot tell them from a larva. They are listed apart on purpose.

Options:   --n 30 (clips per sound, default 30)   --dur 12   --seed 1   --json out.json
"""
import argparse
import json
import sys
import tempfile
import os
import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly

import noise_sounds as ns
import demo_audio
from predict import predict_severity_from_array, load_audio_any
from fusion import fuse_audio_image


def to_wav_4k(x, rng):
    """what try.html does: mono, 4 kHz (anti-aliased), 16-bit WAV, at whatever loudness the microphone gave"""
    y = resample_poly(x, 1, ns.SR0 // 4000)
    y = y / (np.max(np.abs(y)) + 1e-9) * rng.uniform(0.03, 0.7)
    return (np.clip(y, -1, 1) * 32767).astype(np.int16)


def run_one(gen, dur, rng):
    x = gen(dur, rng)
    wav = to_wav_4k(x, rng)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
        path = f.name
    wavfile.write(path, 4000, wav)
    try:
        samples, sr = load_audio_any(path)          # the same reader the server uses
        return demo_audio.analyze_clip(samples, sr, predict_severity_from_array, fuse_audio_image)
    finally:
        os.remove(path)


def run_group(name, gens, n, dur, seed):
    rows = {}
    for k, (label, gen) in enumerate(gens.items()):
        rng = np.random.default_rng(seed * 1000 + k * 97)
        res = [run_one(gen, dur, rng) for _ in range(n)]
        shown = sum(r["stage"] != "healthy" for r in res)                      # what the visitor would be told
        model_only = sum(r["model_stage"] != "healthy" for r in res)           # what the model alone said
        inconclusive = sum(bool(r.get("inconclusive")) for r in res)
        pulses = [r["evidence"]["pulses"] for r in res]
        rows[label] = {"n": n, "shown_as_infested": shown, "model_alone_infested": model_only, "inconclusive": inconclusive,
                       "median_pulses": float(np.median(pulses)), "max_pulses": int(max(pulses))}
    return rows


def table(title, rows, want_infested):
    print(f"\n{title}")
    print(f"{'sound':<18}{'n':>4}{'model alone':>14}{'site shows':>13}{'inconcl.':>10}{'median pulses':>15}")
    for k, r in rows.items():
        print(f"{k:<18}{r['n']:>4}{r['model_alone_infested']:>14}{r['shown_as_infested']:>13}{r['inconclusive']:>10}{r['median_pulses']:>15.1f}")
    tot = sum(r["n"] for r in rows.values())
    shown = sum(r["shown_as_infested"] for r in rows.values())
    model = sum(r["model_alone_infested"] for r in rows.values())
    if want_infested:
        print(f"  -> heard as infested by the site: {shown}/{tot} ({shown / tot:.0%});  by the model alone: {model}/{tot} ({model / tot:.0%})")
    else:
        print(f"  -> FALSE ALARMS shown by the site: {shown}/{tot} ({shown / tot:.1%});  by the model alone (no pulse check): {model}/{tot} ({model / tot:.1%})")
    return shown, tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--dur", type=float, default=12.0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    out = {}
    fa, ft = table("A) sounds that are NOT weevils (air, wind on the microphone, breath, fan, hum, speech, crowd, rustle ...) - must NOT say infested",
                   out.setdefault("not_weevil", run_group("not_weevil", {**ns.NOT_WEEVIL, **ns.SPEAKER_NOT_WEEVIL}, a.n, a.dur, a.seed)), False)
    table("B) short sharp taps that are NOT larvae (the honest limit: they ARE short pulses)",
          out.setdefault("short_taps", run_group("short_taps", ns.SHORT_NOT_LARVAE, a.n, a.dur, a.seed)), False)
    wa, wt = table("C) simulated weevil sounds, direct and played from a phone speaker in a room - should still be heard",
                   out.setdefault("weevil", run_group("weevil", {**ns.WEEVIL, **ns.SPEAKER_WEEVIL}, a.n, a.dur, a.seed)), True)
    out["summary"] = {"false_alarms": fa, "not_weevil_clips": ft, "weevil_heard": wa, "weevil_clips": wt}
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
    print(f"\nSUMMARY  false alarms {fa}/{ft} ({fa / ft:.1%})   |   simulated weevil heard {wa}/{wt} ({wa / wt:.0%})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
