"""
مولّد بيانات صوتية محاكاة - حفار ساق النخيل ذو القرون الطويلة
(Jebusaea hammerschmidti)  ⚠️ تجريبي - افتراضات مو قياسات حقيقية
==================================================================
تنبيه صريح: ما وجدنا أي تسجيل صوتي منشور لهذي الآفة. كل معاملات هذا
المولّد (المدى الترددي، مدة النبضة، معدل التكرار) هي *افتراضات* بُنيت على
فكرة عامة إن يرقات الخنافس طويلة القرون كبيرة الحجم وتقرض الخشب بنبضات
أطول وأعلى طاقة من يرقات السوسة. هذي فرضية للاختبار، مو حقيقة علمية.

الاستخدام المسموح:  اختبار أن خط المعالجة (features -> RandomForest) يشتغل
                     مع نوع آفة ثاني، وكمسودة عمل مستقبلي.
الاستخدام الممنوع:   نسبة أي دقة لهذا النموذج، أو الادعاء بأن النظام يكشف
                     هذي الآفة، قبل استبدال هذي البيانات بتسجيلات ميدانية.

متوافق مع نفس صيغة generate_synthetic_data.py (ملفات .npy + manifest.json
بنفس الفئات الأربع)، فيمكن توجيه train_model.py لمجلده بدون تغيير آخر.

قيد تقني: معدل العينات 4000Hz يعني أقصى تردد نلتقطه ~2000Hz فقط.
"""
import numpy as np
import os
import json

SAMPLE_RATE = 4000
DURATION_SEC = 5
CLASSES = ["سليم", "مبكرة", "متوسطة", "متقدمة"]
CLASS_IDS = {c: i for i, c in enumerate(CLASSES)}

# افتراض: يرقة أكبر -> نبضات أقل تكرارًا لكن أقوى وأطول من السوسة
PULSE_RATE = {"سليم": 0.0, "مبكرة": 0.35, "متوسطة": 1.0, "متقدمة": 2.4}
PULSE_AMP = {"سليم": 0.0, "مبكرة": 0.3, "متوسطة": 0.55, "متقدمة": 0.8}
PULSE_LEN_SEC = 0.12        # السوسة بالمولّد الأصلي: 0.05
FREQ_RANGE = (400, 1400)    # السوسة بالمولّد الأصلي: 200-500 (افتراض)


def pink_noise(n_samples, rng):
    white = rng.normal(0, 1, n_samples)
    fft = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n_samples, d=1 / SAMPLE_RATE)
    freqs[0] = freqs[1]
    fft = fft / np.sqrt(freqs)
    pink = np.fft.irfft(fft, n=n_samples)
    return pink / (np.max(np.abs(pink)) + 1e-9)


def borer_pulse(t_rel, freq):
    """نبضة قرض أطول (انحلال أبطأ) مع مركّبة توافقية ثانية تخلي الصوت أخشن."""
    envelope = np.exp(-t_rel * 18) * (t_rel >= 0)
    return envelope * (np.sin(2 * np.pi * freq * t_rel)
                       + 0.4 * np.sin(2 * np.pi * 1.7 * freq * t_rel))


def generate_sample(cls, rng):
    n = int(SAMPLE_RATE * DURATION_SEC)
    t = np.arange(n) / SAMPLE_RATE
    background = 0.15 * pink_noise(n, rng)
    signal = np.zeros(n)

    rate, amp = PULSE_RATE[cls], PULSE_AMP[cls]
    if rate > 0:
        n_pulses = rng.poisson(rate * DURATION_SEC)
        pulse_times = rng.uniform(0, DURATION_SEC - PULSE_LEN_SEC, size=max(n_pulses, 0))
        freq_center = rng.uniform(*FREQ_RANGE)
        for pt in pulse_times:
            idx0 = int(pt * SAMPLE_RATE)
            idx1 = min(idx0 + int(PULSE_LEN_SEC * SAMPLE_RATE), n)
            t_rel = t[idx0:idx1] - pt
            signal[idx0:idx1] += amp * rng.uniform(0.7, 1.3) * borer_pulse(t_rel, freq_center)

    audio = background * rng.uniform(0.85, 1.15) + signal
    audio = audio / (np.max(np.abs(audio)) + 1e-9) * 0.9
    return audio.astype(np.float32)


def build_dataset(n_per_class=60, seed=7, out_dir=None):
    if out_dir is None:
        out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_raw_borer")
    rng = np.random.default_rng(seed)
    os.makedirs(out_dir, exist_ok=True)
    manifest = []
    for cls in CLASSES:
        for i in range(n_per_class):
            fname = f"{cls}_{i:03d}.npy"
            np.save(os.path.join(out_dir, fname), generate_sample(cls, rng))
            manifest.append({"file": fname, "label": cls, "label_id": CLASS_IDS[cls]})
    # manifest.json يبقى قائمة عادية (نفس صيغة مولّد السوسة) عشان يظل متوافق
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    with open(os.path.join(out_dir, "WARNING.txt"), "w", encoding="utf-8") as f:
        f.write("SYNTHETIC, ASSUMPTION-BASED DATA - NOT REAL RECORDINGS.\n"
                "Do not attribute accuracy to any model trained on this folder.\n")
    print(f"generated {len(manifest)} synthetic borer samples in {out_dir} (ASSUMPTION-BASED)")
    return manifest


if __name__ == "__main__":
    build_dataset()
