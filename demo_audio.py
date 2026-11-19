"""
Analysis of one recording for the public try page (nothing is stored, no Gemini).

Two steps, and both are shown to the visitor:
 1. the trained 4-stage model looks at the recording (every 5 s window of it, strongest window wins);
 2. a plain pulse check (acoustic_check.py) asks whether short, separate, abrupt pulses that fall back to quiet were really
    heard. If the model says "infested" but no pulses were heard (air, breath, a fan, speech...), the verdict is lowered to
    "no sign" and the page says so. If the recording itself is the problem (air or rumble on the microphone, an overloaded
    microphone, silence) the page says "could not judge, record again" instead of "no sign". The pulse check can only lower
    an alarm, never raise one. verify_noise.py measures how often non-weevil sounds still get through.
"""
import numpy as np
from acoustic_check import pulse_evidence, has_pulses, min_events_for, cannot_judge

WINDOW_S = 5.0     # the model was trained on 5 s clips
STEP_S = 2.5
MAX_S = 15.0       # longer recordings are cut to this

HEALTHY = "سليم"
STAGE_KEYS = {"سليم": "healthy", "مبكرة": "early", "متوسطة": "moderate", "متقدمة": "advanced"}


def _norm(x):
    return (x / (np.max(np.abs(x)) + 1e-9) * 0.9).astype(np.float32)


def _windows(n, sr):
    w, st = int(WINDOW_S * sr), int(STEP_S * sr)
    if n <= w + st // 2:
        return [(0, n)]
    out, s = [], 0
    while s + w <= n:
        out.append((s, s + w))
        s += st
    if out[-1][1] < n - st // 2:
        out.append((n - w, n))
    return out


def analyze_clip(audio, sr, predict_fn, fuse_fn, clipped=False):
    """audio: float array at sr. predict_fn = predict.predict_severity_from_array, fuse_fn = fusion.fuse_audio_image.
    clipped: True when the browser saw the microphone overloaded (flat-topped samples) before it made the 4 kHz copy.
    Returns a dict ready for JSON (stage keys are English: healthy/early/moderate/advanced).
    "inconclusive": True means the recording itself is the problem (almost silent, or air / rumble / an overloaded microphone):
    the page then says "record again" and does not show a stage."""
    x = np.asarray(audio, dtype=np.float32).ravel()
    x = x[: int(MAX_S * sr)]
    ev = pulse_evidence(x, sr)
    if clipped:                      # the browser saw flat-topped samples before it made the small 4 kHz copy (the copy hides them)
        ev["clipped"] = True
    # pick the window with the most pulses (ties: the first); one window when the clip is short
    best, best_n = None, -1
    for a, b in _windows(len(x), sr):
        n = pulse_evidence(x[a:b], sr)["n_events"] if len(x) > int(WINDOW_S * sr) + 1 else ev["n_events"]
        if n > best_n:
            best, best_n = (a, b), n
    a, b = best
    audio_result = predict_fn(_norm(x[a:b]), sr)
    fusion = fuse_fn(audio_result, None)
    cls = fusion["final_class"]
    pulses_ok = has_pulses(ev)
    reason = None if pulses_ok else cannot_judge(ev)         # 'air' / 'silent': the recording itself is the problem, so "no sign" would be the wrong thing to say
    inconclusive = reason is not None
    vetoed = (cls != HEALTHY) and not pulses_ok and not inconclusive
    probs = {STAGE_KEYS.get(k, k): float(v) for k, v in audio_result["probabilities"].items()}
    out = {
        "stage": "healthy" if (vetoed or inconclusive) else STAGE_KEYS.get(cls, "healthy"),
        "model_stage": STAGE_KEYS.get(cls, "healthy"),
        "vetoed": bool(vetoed),
        "inconclusive": bool(inconclusive),
        "reason": reason,
        "confidence": None if (vetoed or inconclusive) else float(fusion.get("confidence") if fusion.get("confidence") is not None else audio_result["confidence"]),
        "risk": None if (vetoed or inconclusive) else float(fusion["final_score"]),
        "probabilities": probs,
        "noise_filter": audio_result.get("noise_filter"),
        "evidence": {
            "pulses": int(ev["n_events"]), "per_s": float(ev["events_per_s"]), "duration_s": float(ev["duration_s"]),
            "needed": int(min_events_for(ev["duration_s"])), "times_s": ev.get("times_s", []),
            "sustained": bool(ev.get("active_fraction", 0.0) > 0.35), "silent": bool(ev.get("silent")),
            "crackle": bool(ev.get("crackle")), "air_on_mic": bool(ev.get("air_on_mic")), "clipped": bool(ev.get("clipped")),
            "rumble_ratio": float(ev.get("rumble_ratio", 0.0)),
        },
        "window": [round(a / sr, 1), round(b / sr, 1)],
    }
    return out
