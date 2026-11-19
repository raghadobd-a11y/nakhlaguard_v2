"""
Make 4 demo WAV files (one per stage) and TEST them with YOUR trained model.   (ASCII-only file)
Run from the project folder, after  python train_model.py :
    python make_demo_wavs.py
Output: folder  demo_wavs  with  1_healthy.wav  2_early.wav  3_moderate.wav  4_advanced.wav
and a table with the model's REAL answer for each one. Use these files in the live demo.

They are SIMULATED recordings made by our own generator (the same one used for training).
For each stage we try 60 random clips and keep the clearest one the model gets right (and, for the infested stages, that really contains 3 or more pulses),
so say "simulated test recording" when you present them.
"""
import os
import numpy as np
from scipy.io import wavfile
from generate_synthetic_data import generate_sample, CLASSES, SAMPLE_RATE
from predict import predict_severity_from_array, predict_severity_from_file
from acoustic_check import pulse_evidence, has_pulses

NAMES = ["1_healthy", "2_early", "3_moderate", "4_advanced"]
OUT = "demo_wavs"
os.makedirs(OUT, exist_ok=True)
print(f"{'file':<18}{'predicted':<12}{'confidence':<12}  correct among 60 tries")
for k, cls in enumerate(CLASSES):
    rng = np.random.default_rng(1000 + k)
    best, ok = None, 0
    for _ in range(60):
        a = generate_sample(cls, rng)
        r = predict_severity_from_array(a, SAMPLE_RATE)
        if r["predicted_class"] == cls:
            ok += 1
            ev = pulse_evidence(a, SAMPLE_RATE)
            # the website also runs a pulse check: a non-healthy sample must really contain pulses (3 or more), or the page would lower it to "healthy"
            if cls != CLASSES[0] and not (has_pulses(ev) and ev["n_events"] >= 3):
                continue
            if best is None or r["confidence"] > best[1]:
                best = (a, r["confidence"])
    if best is None:
        print(f"{NAMES[k]:<18}-- no clip was classified correctly; retrain the model and run again"); continue
    path = os.path.join(OUT, NAMES[k] + ".wav")
    wavfile.write(path, SAMPLE_RATE, (best[0] * 32767).astype(np.int16))
    r2 = predict_severity_from_file(path)        # re-check from the saved file, exactly what the website will do
    print(f"{NAMES[k]:<18}{r2['predicted_class']:<12}{r2['confidence']*100:>5.0f}%       {ok}/60")
print("\nDone. Upload these files on the test page (test.html). The Arabic class names above are the model's own labels.")
