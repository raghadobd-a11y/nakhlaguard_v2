"""
Make three SIMULATED sounds you can play from a phone or speaker in front of the microphone, to try the "Try" page live:
    weevil_dense_sim.wav      many chewing pulses (clear infestation)
    weevil_sparse_sim.wav     a few pulses (the start of an infestation)
    air_noise_sim.wav         only air / room noise (control: the page must say "no sign")
They are made by our own generator (damped 400-560 Hz pulses on a pink-noise background, the same kind of signal the model
was trained on), 12 seconds each, 16 kHz mono. They are NOT recordings from a real palm. Say so when you use them.
Run:  python make_weevil_sound.py      (output goes to the folder demo_wavs)
"""
import os
import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, sosfilt

SR = 16000
DUR = 12.0


def pink(n, rng):
    f = np.fft.rfft(rng.normal(size=n))
    fr = np.fft.rfftfreq(n, 1 / SR); fr[0] = fr[1]
    p = np.fft.irfft(f / np.sqrt(fr), n=n)
    return p / (np.max(np.abs(p)) + 1e-9)


def pulse(length, freq, decay):
    t = np.arange(length) / SR
    env = np.exp(-t * decay)
    return env * np.sin(2 * np.pi * freq * t) + 0.35 * env * np.sin(2 * np.pi * 2.1 * freq * t)   # a little crackle on top


def make(rate, amp, seed, bg=0.12):
    rng = np.random.default_rng(seed)
    n = int(SR * DUR)
    x = bg * pink(n, rng)
    if rate > 0:
        # activity comes in short bursts (a few pulses close together), as the literature describes, then pauses
        t = 0.3
        while t < DUR - 0.2:
            k = rng.integers(2, 5)
            for j in range(k):
                i = int((t + j * rng.uniform(0.07, 0.16)) * SR)
                L = int(0.06 * SR)
                if i + L < n:
                    x[i:i + L] += amp * rng.uniform(0.75, 1.25) * pulse(L, rng.uniform(400, 560), 55)
            t += k / rate * rng.uniform(0.8, 1.2)
    x = x / (np.max(np.abs(x)) + 1e-9) * 0.9
    return x.astype(np.float32)


def air(seed=5):
    rng = np.random.default_rng(seed)
    n = int(SR * DUR); t = np.arange(n) / SR
    w = rng.normal(size=n)
    low = sosfilt(butter(2, 900, 'low', fs=SR, output='sos'), w)
    gust = 0.6 + 0.4 * np.sin(2 * np.pi * 0.35 * t + 1.0)
    x = 0.8 * low * gust + 0.25 * pink(n, rng)
    return (x / (np.max(np.abs(x)) + 1e-9) * 0.5).astype(np.float32)


def write(path, x):
    wavfile.write(path, SR, (x * 32767).astype(np.int16))


if __name__ == "__main__":
    out = "demo_wavs"
    os.makedirs(out, exist_ok=True)
    write(os.path.join(out, "weevil_dense_sim.wav"), make(rate=2.6, amp=0.9, seed=21))
    write(os.path.join(out, "weevil_sparse_sim.wav"), make(rate=0.6, amp=0.6, seed=22))
    write(os.path.join(out, "air_noise_sim.wav"), air())
    print("Done: demo_wavs/weevil_dense_sim.wav, weevil_sparse_sim.wav, air_noise_sim.wav")
