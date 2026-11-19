"""
الاستدلال (Inference) - مشروع "نخلة"
====================================
يحمّل النموذج المدرّب ويصنّف تسجيل صوتي جديد إلى إحدى فئات شدة الإصابة،
مع نسبة ثقة لكل فئة (مفيد لواجهة المستخدم ولـ Fusion Score لاحقًا).
"""
import os
import numpy as np
import joblib
from scipy.io import wavfile

from features import extract_features

MODEL_PATH = os.path.join(os.path.dirname(__file__), "nakhla_model.joblib")
_bundle = None

# مرشّح الضجيج (اختياري): كاشف ثنائي (سليم/مصاب) مدرَّب على تسجيلات حقيقية وأصوات بيئية بـ train_treevibes.py.
# إن وُجد الملف وقال إن التسجيل لا يشبه الإصابة، نُنزل أي نتيجة "مصابة" إلى "سليم" فلا يظهر تنبيه. لا يرفع تنبيهًا أبدًا.
# للتعطيل (مثلًا أثناء تجربة تسجيل محاكى): $env:NAKHLA_GATE="off"   ولتغيير العتبة: $env:NAKHLA_GATE_THRESHOLD="0.5" (أقل = حساسية أعلى للإصابة)
BINARY_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nakhla_binary_model.joblib")
_gate, _gate_loaded = None, False


def _load_gate():
    global _gate, _gate_loaded
    if os.environ.get("NAKHLA_GATE", "on").lower() == "off":
        return None
    if not _gate_loaded:
        _gate_loaded = True
        if os.path.exists(BINARY_PATH):
            try:
                _gate = joblib.load(BINARY_PATH)
            except Exception as e:
                print(f"[NakhlaGuard] could not load the noise filter: {type(e).__name__}: {e}")
    return _gate


def _load_bundle():
    global _bundle
    if _bundle is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                "لا يوجد نموذج مدرَّب. شغِّل train_model.py أولًا."
            )
        _bundle = joblib.load(MODEL_PATH)
    return _bundle


def load_audio_any(path, target_sr=4000):
    """يقرأ ملف .wav أو .npy ويعيد موجة صوتية بمعدل العينات المطلوب."""
    if path.endswith(".npy"):
        return np.load(path), target_sr
    sr, audio = wavfile.read(path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float32)
    audio = audio / (np.max(np.abs(audio)) + 1e-9)
    if sr != target_sr:
        # إعادة أخذ عينات مع مرشح لمنع التداخل الترددي (أدق من الاستيفاء الخطي عند تحويل 8 أو 44.1 كيلوهرتز إلى 4)
        from math import gcd
        from scipy.signal import resample_poly
        g = gcd(int(sr), int(target_sr))
        audio = resample_poly(audio, int(target_sr) // g, int(sr) // g)
    return audio.astype(np.float32), target_sr


def predict_severity_from_array(audio, sr=4000):
    bundle = _load_bundle()
    clf, classes = bundle["model"], bundle["classes"]
    feats = extract_features(audio, sr=sr).reshape(1, -1)
    proba = clf.predict_proba(feats)[0]
    pred_idx = int(np.argmax(proba))
    result = {
        "predicted_class": classes[pred_idx],
        "confidence": float(proba[pred_idx]),
        "probabilities": {classes[i]: float(p) for i, p in enumerate(proba)},
    }
    gate = _load_gate()
    if gate is not None:
        if gate.get("features", "base") == "extra":
            from features_extra import extra_features
            gfeats = extra_features(audio, sr).reshape(1, -1)
        else:
            gfeats = feats
        p_inf = float(gate["model"].predict_proba(gfeats)[0, 1])
        healthy = classes[0]
        thr = float(os.environ.get("NAKHLA_GATE_THRESHOLD", gate["threshold"]))   # لتغيير الحساسية بدون إعادة تدريب: $env:NAKHLA_GATE_THRESHOLD="0.5"
        suppressed = p_inf < thr and result["predicted_class"] != healthy
        result["noise_filter"] = {"p_infested": p_inf, "threshold": thr, "suppressed": bool(suppressed)}
        if suppressed:
            others = {c: p for c, p in result["probabilities"].items() if c != healthy}
            tot = sum(others.values()) or 1.0
            result["probabilities"] = {healthy: 1.0 - p_inf, **{c: p / tot * p_inf for c, p in others.items()}}
            result["predicted_class"], result["confidence"] = healthy, 1.0 - p_inf
    return result


def predict_severity_from_file(path):
    audio, sr = load_audio_any(path)
    return predict_severity_from_array(audio, sr=sr)


if __name__ == "__main__":
    import sys
    import glob

    # اختبار سريع على عينات محاكاة موجودة
    raw_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_raw")
    sample_files = sorted(glob.glob(os.path.join(raw_dir, "*.npy")))[:4]
    for f in sample_files:
        result = predict_severity_from_file(f)
        print(os.path.basename(f), "->", result["predicted_class"],
              f"(ثقة {result['confidence']:.0%})")
