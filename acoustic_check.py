"""
Pulse evidence check for the public try page (and anything else that gets a recording from an unknown source).

Why it exists: the 4-stage model was trained on SIMULATED clips. A recording of plain air, breath or room noise is far
from anything it saw, and it can still answer "infested" (in our own test it did so for about 4 out of 5 non-weevil sounds,
see verify_noise.py). Literature (claims L3, L4) describes larval activity as trains of SHORT pulses (milliseconds to a few
tens of ms) that rise above a quiet background and fall back to it. So before we believe a non-healthy answer we ask a simple,
explainable question about every candidate pulse. A pulse only counts when ALL of these hold:

  1. it is well above the local background level (HEIGHT_RATIO times);
  2. it is short (envelope width at half height below MAX_WIDTH_S) - speech syllables, gusts and breaths are longer;
  3. it stands out from its own surroundings: within +-0.3 s the quietest parts are far below the pulse (LOCAL_RATIO times),
     so a loud stretch of noise full of random peaks (breath, blowing, rustle) is not a train of pulses;
  4. it lives in the band where larval sounds are described (150-1800 Hz), not in the thump band below 120 Hz where
     air puffs, wind on the microphone and handling noise sit (LF_MAX_RATIO).

Two more things are measured for the whole clip, because a pulse count alone cannot say them:
  * sustained: the envelope is far above the background for a large share of the time (a continuous sound);
  * crackle: far too many pulses per second to be a larva (rustling leaves, paper, fabric).
  * air_on_mic: low-frequency rumble (20-120 Hz) is much stronger than the band we listen to - the signature of wind,
    breath or handling noise on the microphone. With no pulses this is reported as "cannot judge, record again", not as "healthy".
  * clipped: the microphone was overloaded (blowing straight into it).

This is a plain signal-processing rule (no training). It is NOT proof of infestation, and it does not make the model better
on real palms. It stops air, wind, breath, hum, fans, speech and rustling from being reported as infestation. A short sharp tap
(a fingernail, a key) IS a short pulse and cannot be told from a larva by this check; the page says so.
"""
import math
import numpy as np
from scipy.signal import butter, sosfiltfilt, hilbert, find_peaks, peak_widths

BAND = (150.0, 1800.0)        # Hz. The band where larval sounds are described; below 150 Hz sit wind rumble and mains hum.
LF_BAND = (20.0, 120.0)       # Hz. The thump band: air puffs, wind on the microphone, handling noise.
ENV_HZ = 200.0                # the envelope is kept at 200 values per second (5 ms bins)
FLOOR_WINDOW_S = 1.0          # background level = a low percentile of the envelope in a sliding 1 s window
FLOOR_PCT = 40
HEIGHT_RATIO = 6.0            # a pulse must rise this many times above the local background
MIN_DISTANCE_S = 0.045        # two pulses closer than this count as one
MAX_WIDTH_S = 0.080           # a pulse is short: envelope width at half height below 80 ms
LOCAL_WINDOW_S = 0.12         # look this far to each side of a pulse (outside its own 25 ms) at what is around it
LOCAL_PCT = 50                # the typical (median) level around the pulse
LOCAL_RATIO = 4.0             # the pulse must be this many times above the typical level around it: a noise burst with deep dips is loud all around, a pulse is not
VALLEY_WINDOW_S = 0.10        # on both sides of a pulse, within 0.1 s, the sound must fall back to a quiet level
VALLEY_RATIO = 3.0            # the pulse must be at least this many times above the quietest point on each side
RISE_MAX_S = 0.020            # a pulse starts abruptly: from 20 % to its top in at most 20 ms (a gust or a breath swells slowly)
LF_MAX_RATIO = 1.5           # energy under 120 Hz around the pulse must be below this times the energy in the 150-1800 Hz band
MIN_EVENTS = 2                # fewer separate pulses than this in the whole clip: "no larva-like pulses"
MAX_ACTIVE_FRACTION = 0.35    # if the envelope is far above the background for more than this share of the time it is a sustained sound
MAX_EVENTS_PER_S = 10.0       # more than this is crackle (rustle, paper), not a larva
AIR_RUMBLE_RATIO = 1.2        # clip-level: rumble (20-120 Hz) stronger than this times the 150-1800 Hz band = wind / breath / handling on the microphone
CLIP_FRACTION = 0.002         # more than 0.2 % of the samples on flat tops = overloaded microphone


def _bandpass(x, sr, lo, hi, order=4):
    hi = min(hi, sr / 2 - 50.0)
    sos = butter(order, [lo, hi], btype="bandpass", fs=sr, output="sos")
    return sosfiltfilt(sos, x)


def _envelope(y, sr):
    env = np.abs(hilbert(y))
    step = max(1, int(round(sr / ENV_HZ)))
    n = (len(env) // step) * step
    return env[:n].reshape(-1, step).mean(axis=1), sr / step


def _sliding_pct(env, w, pct):
    w = max(3, int(w)) | 1
    pad = np.pad(env, (w // 2, w // 2), mode="reflect")
    return np.percentile(np.lib.stride_tricks.sliding_window_view(pad, w), pct, axis=1)[: len(env)]


def pulse_evidence(audio, sr=4000):
    """Returns {"n_events", "events_per_s", "duration_s", "median_width_ms", "active_fraction", "silent", "times_s",
    "candidates", "rumble_ratio", "air_on_mic", "clipped", "crackle"}. audio: 1-D float array (already peak-normalised or not)."""
    x = np.asarray(audio, dtype=np.float64).ravel()
    dur = len(x) / float(sr) if sr else 0.0
    out = {"n_events": 0, "events_per_s": 0.0, "duration_s": round(dur, 2), "median_width_ms": None, "active_fraction": 0.0,
           "silent": False, "times_s": [], "candidates": 0, "rumble_ratio": 0.0, "air_on_mic": False, "clipped": False, "crackle": False}
    if len(x) < sr * 0.5:
        out["silent"] = True
        return out
    x = x - np.mean(x)
    if float(np.sqrt(np.mean(x ** 2))) < 1e-5:
        out["silent"] = True
        return out
    top = float(np.max(np.abs(x)))
    m = np.abs(x) >= 0.995 * top                      # clipping = flat tops: three or more samples in a row at the very top
    out["clipped"] = bool(np.mean(m & np.roll(m, 1) & np.roll(m, -1)) > CLIP_FRACTION)
    y = _bandpass(x, sr, BAND[0], BAND[1])
    lf = _bandpass(x, sr, LF_BAND[0], LF_BAND[1], order=2)
    out["rumble_ratio"] = round(float(np.sqrt(np.mean(lf ** 2)) / (np.sqrt(np.mean(y ** 2)) + 1e-12)), 3)
    out["air_on_mic"] = bool(out["rumble_ratio"] > AIR_RUMBLE_RATIO)
    env, er = _envelope(y, sr)
    floor = _sliding_pct(env, FLOOR_WINDOW_S * er, FLOOR_PCT)
    floor = np.maximum(floor, 1e-9 + 0.02 * float(np.median(env)))
    ratio = env / floor
    out["active_fraction"] = round(float(np.mean(ratio > 2.5)), 3)
    peaks, _ = find_peaks(ratio, height=HEIGHT_RATIO, distance=max(1, int(MIN_DISTANCE_S * er)))
    out["candidates"] = int(len(peaks))
    if len(peaks) == 0:
        return out
    widths = peak_widths(ratio, peaks, rel_height=0.5)[0] / float(er)
    lf_env, _ = _envelope(lf, sr)
    fine = np.abs(hilbert(y))                       # 1 ms bins, to measure how abruptly a pulse starts
    fstep = max(1, int(round(sr / 1000.0)))
    fine = fine[: (len(fine) // fstep) * fstep].reshape(-1, fstep).mean(axis=1)
    k5 = max(1, int(round(0.005 * sr / fstep)))      # smooth over 5 ms so that the jitter of noise does not look like an abrupt start
    fine = np.convolve(fine, np.ones(k5) / k5, mode="same")
    f_per_bin = (sr / fstep) / er                    # fine bins per coarse bin (5)
    half = max(2, int(LOCAL_WINDOW_S * er))
    guard = max(1, int(0.025 * er))
    vwin = max(2, int(VALLEY_WINDOW_S * er))
    keep = np.zeros(len(peaks), dtype=bool)
    for k, p in enumerate(peaks):
        if widths[k] > MAX_WIDTH_S:
            continue
        left, right = env[max(0, p - vwin):p], env[p + 1:p + 1 + vwin]
        if len(left) < 2 or len(right) < 2:
            continue                      # at the very edge of the clip: cannot see both sides of it
        if env[p] < VALLEY_RATIO * max(float(np.min(left)), float(np.min(right)), 1e-9):
            continue                      # the sound does not fall back after (or before) it: a wiggle on a swell of noise, not a pulse
        a, b = max(0, p - half), min(len(env), p + half + 1)
        around = np.concatenate([env[a:max(a, p - guard)], env[min(b, p + guard + 1):b]])
        if len(around) < 4:
            continue
        quiet = float(np.percentile(around, LOCAL_PCT))
        if env[p] < LOCAL_RATIO * max(quiet, 1e-9):
            continue                      # it does not stand out from its surroundings: a loud stretch of noise, not a pulse
        s, e = max(0, p - guard), min(len(env), p + 3 * guard)
        if float(np.mean(lf_env[s:e])) > LF_MAX_RATIO * float(np.mean(env[s:e])):
            continue                      # a low thump (air puff, handling): its energy sits under 120 Hz
        q0 = int(p * f_per_bin)
        seg = fine[max(0, q0 - int(2 * f_per_bin)): q0 + int(3 * f_per_bin)]
        if len(seg) < 3:
            continue
        qpk = max(0, q0 - int(2 * f_per_bin)) + int(np.argmax(seg))
        base = float(np.min(fine[max(0, qpk - 100):qpk + 1]))
        top_v = float(fine[qpk])
        lvl = base + 0.2 * (top_v - base)
        j = qpk
        while j > max(0, qpk - 100) and fine[j] > lvl:
            j -= 1
        if (qpk - j) / 1000.0 > RISE_MAX_S:
            continue                      # it swells slowly: wind, breath or handling, not a sharp strike
        keep[k] = True
    n = int(np.sum(keep))
    out["n_events"] = n
    out["events_per_s"] = round(n / dur, 2) if dur else 0.0
    out["crackle"] = bool(out["events_per_s"] > MAX_EVENTS_PER_S)
    out["times_s"] = [round(float(t) / er, 2) for t in peaks[keep][:80]]
    if n:
        out["median_width_ms"] = round(float(np.median(widths[keep])) * 1000.0, 1)
    return out


def min_events_for(duration_s):
    """At least MIN_EVENTS pulses, and for long clips about one pulse per 5 seconds (so 12 s needs 3)."""
    return max(MIN_EVENTS, int(math.ceil(0.2 * float(duration_s or 0.0))))


def has_pulses(evidence):
    """True only when enough separate larva-like pulses were heard AND the recording is not dominated by air on the microphone,
    a continuous sound, or crackle."""
    return ((not evidence.get("silent")) and evidence.get("n_events", 0) >= min_events_for(evidence.get("duration_s"))
            and evidence.get("active_fraction", 0.0) <= MAX_ACTIVE_FRACTION and not evidence.get("crackle", False)
            and not evidence.get("air_on_mic", False))


def cannot_judge(evidence):
    """Reason string when the recording itself is the problem (so "no sign" would be the wrong message), else None.
    'silent' = almost no sound; 'air' = wind / breath / handling rumble or an overloaded microphone on top of everything."""
    if evidence.get("silent"):
        return "silent"
    if evidence.get("air_on_mic") or evidence.get("clipped"):
        return "air"
    return None
