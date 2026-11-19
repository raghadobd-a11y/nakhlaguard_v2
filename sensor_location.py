"""
Location of the vibration source on the trunk from three sensors (server side).   (ASCII-only)
Same model and numbers as the browser page localize.html, so both agree:  log(A_i) = c - alpha*d_i - beta*log(d_i).
Input: the average level (RMS) of the three channels, and optionally a quiet baseline to subtract.
Output: x, y in cm on the 40 x 80 cm sensor panel, or None when the signal is too weak to trust.

HONEST LIMITS: simulation-tested only. Without a calibration file the attenuation (alpha) is a default and the position is NOT calibrated,
so the result is marked mode="live" (uncalibrated). A file sensor_calibration.json next to this module can hold {"alpha": ..., "gains": [..3..], "error_cm": ...}
measured with known taps; then mode="cal" and error_cm is reported as the uncertainty.
"""
import os
import json
import math
import threading
from datetime import datetime

import numpy as np

W, H = 40, 80
SENSORS = np.array([[8.0, 14.0], [32.0, 14.0], [20.0, 66.0]])
BETA, NOISE, ALPHA_DEFAULT, THRESHOLD = 0.5, 0.6, 0.05, 4.0
HERE = os.path.dirname(os.path.abspath(__file__))
CAL_PATH = os.path.join(HERE, "sensor_calibration.json")
STORE_PATH = os.path.join(HERE, "sensor_locations.json")
_LOCK = threading.Lock()

_GX, _GY = np.meshgrid(np.arange(0, W + 1, dtype=float), np.arange(0, H + 1, dtype=float), indexing="ij")
_D = [np.hypot(_GX - sx, _GY - sy) + 1.0 for sx, sy in SENSORS]
_MODEL = [-0.0 - BETA * np.log(d) for d in _D]          # the alpha-free part of the expected log-level


def estimate(sig, alpha=ALPHA_DEFAULT, gains=(0.0, 0.0, 0.0)):
    sig = [float(v) for v in sig]
    logs = [math.log(max(v, NOISE)) - float(gains[i]) for i, v in enumerate(sig)]
    w = [min(1.0, max(v - NOISE, 0.0) / (3 * NOISE)) for v in sig]
    sw = sum(w)
    if sw < 0.3:
        return None
    if sum(1 for v in w if v >= 0.5) < 3:                      # fewer than 3 strong channels: rough estimate near the strongest sensor
        q = [v * v for v in sig]; qs = sum(q)
        return {"x": sum(SENSORS[i, 0] * q[i] for i in range(3)) / qs, "y": sum(SENSORS[i, 1] * q[i] for i in range(3)) / qs, "err": 0.0, "coarse": True}
    diff = [logs[i] - (-alpha * _D[i] + _MODEL[i]) for i in range(3)]
    m = (w[0] * diff[0] + w[1] * diff[1] + w[2] * diff[2]) / sw
    err = w[0] * (diff[0] - m) ** 2 + w[1] * (diff[1] - m) ** 2 + w[2] * (diff[2] - m) ** 2
    k = int(np.argmin(err))                                      # first minimum in x-major order, like the browser loops
    x, y = np.unravel_index(k, err.shape)
    return {"x": float(x), "y": float(y), "err": float(err[x, y]), "coarse": False}


def _calibration():
    try:
        with open(CAL_PATH, encoding="utf-8") as f:
            c = json.load(f)
        if "alpha" in c and len(c.get("gains", [])) == 3:
            return c
    except Exception:
        pass
    return None


def locate_from_rms(rms, base=None):
    """rms: 3 average channel levels; base: 3 quiet-baseline levels (optional). Returns a dict or None."""
    try:
        rms = [float(v) for v in rms]; base = [float(v) for v in (base or [0, 0, 0])]
        if len(rms) != 3 or len(base) != 3:
            return None
    except (TypeError, ValueError):
        return None
    sig = [max(r - b, 0.0) for r, b in zip(rms, base)]
    if sum(sig) < THRESHOLD:
        return None
    cal = _calibration()
    e = estimate(sig, cal["alpha"] if cal else ALPHA_DEFAULT, cal["gains"] if cal else (0.0, 0.0, 0.0))
    if e is None:
        return None
    return {"x": round(e["x"], 1), "y": round(e["y"], 1), "coarse": bool(e["coarse"]), "mode": "cal" if cal else "live",
            "err": float(cal.get("error_cm", 0.0)) if cal else 0.0, "levels": [round(v, 2) for v in sig]}


def locate_from_json(text):
    """The server receives the sensor levels as a JSON string {"rms": [..], "base": [..]}. Returns the location or None."""
    if not text:
        return None
    try:
        d = json.loads(text)
        return locate_from_rms(d.get("rms"), d.get("base"))
    except Exception:
        return None


def _read_all():
    try:
        with open(STORE_PATH, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_latest(palm_id, loc):
    with _LOCK:
        d = _read_all()
        d[str(palm_id)] = {"location": loc, "time": datetime.utcnow().isoformat()}
        tmp = STORE_PATH + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(d, f)
        os.replace(tmp, STORE_PATH)


def get_latest(palm_id):
    return _read_all().get(str(palm_id), {"location": None, "time": None})


def demo_location(key):
    """DEMONSTRATION ONLY. Used when no sensors sent any levels (e.g. an uploaded sample recording at an exhibition).
    A source point is chosen from `key` (the same palm always gives the same point) and the three sensor levels are
    SIMULATED from it with the same forward model used in the simulation tests. The point is returned as it is, always
    marked mode="demo": it is NOT a measurement and says nothing about where a larva really is."""
    import hashlib, random
    h = int(hashlib.sha256(str(key).encode("utf-8")).hexdigest()[:12], 16)
    rnd = random.Random(h)
    sx, sy = rnd.uniform(8, 32), rnd.uniform(14, 66)
    levels = []
    for i in range(3):
        d = math.hypot(sx - SENSORS[i, 0], sy - SENSORS[i, 1]) + 1.0
        a = math.exp(7.0 - ALPHA_DEFAULT * d - BETA * math.log(d))
        levels.append(round(max(a * (1.0 + rnd.uniform(-0.03, 0.03)), 0.0), 2))
    return {"x": round(sx, 1), "y": round(sy, 1), "coarse": False, "mode": "demo", "err": 0.0, "levels": levels}
