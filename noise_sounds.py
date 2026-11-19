"""
Synthetic NON-weevil sounds (and weevil look-alikes) for testing the "Record now" check of the public try page.

Everything here is made by code, nothing is a recording from a real microphone or a real palm. They are rough stand-ins for the things
that really happen when a visitor records next to a stand: air on the microphone, breath, a fan, a room, speech, rustling, taps.
Each generator returns a float array at SR0 = 16000 Hz; verify_noise.py converts it to 4 kHz exactly like the web page does.
"""
import numpy as np
from scipy.signal import butter, sosfilt

SR0 = 16000


def _t(dur):
    n = int(SR0 * dur)
    return n, np.arange(n) / SR0


def _white(n, rng):
    return rng.normal(size=n)


def _pink(n, rng):
    f = np.fft.rfft(rng.normal(size=n))
    fr = np.fft.rfftfreq(n, 1 / SR0)
    fr[0] = fr[1]
    return np.fft.irfft(f / np.sqrt(fr), n=n)


def _bp(x, lo, hi, order=2):
    return sosfilt(butter(order, [lo, hi], "bandpass", fs=SR0, output="sos"), x)


def _lp(x, hi, order=2):
    return sosfilt(butter(order, hi, "lowpass", fs=SR0, output="sos"), x)


def _hp(x, lo, order=2):
    return sosfilt(butter(order, lo, "highpass", fs=SR0, output="sos"), x)


def _smooth_env(n, rng, rate_hz, depth, floor=0.15):
    """slowly varying random loudness (gusts): values between `floor` and 1"""
    k = max(4, int(rate_hz * n / SR0) + 2)
    pts = np.abs(rng.normal(size=k)) ** 1.5
    pts = pts / (pts.max() + 1e-9)
    env = np.interp(np.linspace(0, k - 1, n), np.arange(k), pts)
    return floor + (1 - floor) * (env ** (1 + depth))


def _impulse(length, freq, decay, rng=None, noisy=0.0):
    t = np.arange(length) / SR0
    env = np.exp(-t * decay)
    y = env * np.sin(2 * np.pi * freq * t)
    if noisy and rng is not None:
        y = (1 - noisy) * y + noisy * env * rng.normal(size=length)
    return y


# ---------------------------------------------------------------- things that are NOT weevils

def white_room(dur, rng):
    n, _ = _t(dur)
    return _white(n, rng)


def pink_room(dur, rng):
    n, _ = _t(dur)
    return _pink(n, rng)


def mains_hum(dur, rng):
    n, t = _t(dur)
    x = sum(a * np.sin(2 * np.pi * f * t + rng.uniform(0, 6)) for f, a in [(50, 1.0), (100, 0.6), (150, 0.35), (250, 0.2)])
    return x + 0.1 * _pink(n, rng)


def fan(dur, rng):
    n, t = _t(dur)
    base = _lp(_white(n, rng), 1800)
    blade = 0.25 * np.sin(2 * np.pi * rng.uniform(90, 140) * t)
    flutter = 1 + 0.15 * np.sin(2 * np.pi * rng.uniform(0.5, 2.0) * t)
    return (base + blade) * flutter


def air_gusts(dur, rng):
    """moving air: low-passed noise whose loudness rises and falls in gusts"""
    n, _ = _t(dur)
    return _lp(_white(n, rng), rng.uniform(250, 700)) * _smooth_env(n, rng, rate_hz=rng.uniform(0.4, 1.5), depth=1.0)


def air_on_mic(dur, rng):
    """wind hitting a microphone: heavy low-frequency rumble, random turbulent bursts with sharp edges"""
    n, _ = _t(dur)
    rumble = _lp(_white(n, rng), 120, order=2) * 4.0
    turb = _bp(_white(n, rng), 80, 900) * _smooth_env(n, rng, rate_hz=rng.uniform(3, 8), depth=2.0, floor=0.05)
    return (rumble * _smooth_env(n, rng, 0.8, 1.0)) + turb


def mic_pops(dur, rng):
    """air puffs or thumps on the microphone: short low-frequency thumps, 1 to 4 per second, over a quiet room"""
    n, _ = _t(dur)
    x = 0.05 * _pink(n, rng)
    t = 0.2
    while t < dur - 0.2:
        i, L = int(t * SR0), int(0.09 * SR0)
        if i + L < n:
            x[i:i + L] += rng.uniform(0.5, 1.5) * _impulse(L, rng.uniform(40, 110), rng.uniform(40, 90), rng, 0.15)
        t += rng.uniform(0.25, 1.0)
    return x


def breathing(dur, rng):
    """slow breathing close to the microphone: 0.7 s breaths every 2.5-4 s"""
    n, t = _t(dur)
    x = np.zeros(n)
    c = 0.3
    while c < dur:
        L = int(rng.uniform(0.5, 0.9) * SR0)
        i = int(c * SR0)
        if i + L < n:
            w = np.hanning(L) ** 1.5
            x[i:i + L] += w * _bp(_white(L, rng), 200, 3000)
        c += rng.uniform(2.5, 4.0)
    return x + 0.03 * _pink(n, rng)


def blowing(dur, rng):
    """blowing air at the microphone on purpose: strong, irregular, with fast sub-bursts"""
    n, _ = _t(dur)
    x = _bp(_white(n, rng), 100, 2500) * _smooth_env(n, rng, rate_hz=rng.uniform(5, 12), depth=2.5, floor=0.03)
    return x + 0.5 * _lp(_white(n, rng), 100) * _smooth_env(n, rng, 1.0, 1.0)


def blowing_loud(dur, rng):
    """blowing hard into the microphone: the signal is overloaded and clipped flat"""
    x = blowing(dur, rng)
    return np.clip(x / (np.sqrt(np.mean(x ** 2)) + 1e-9) * 2.5, -1.0, 1.0)


def speech_like(dur, rng):
    """made-up speech: voiced vowels (120-220 Hz pulse train through formants) in syllables, with short consonant bursts"""
    n, t = _t(dur)
    f0 = rng.uniform(110, 220)
    glottal = np.zeros(n)
    glottal[(np.arange(0, n, int(SR0 / f0)))] = 1.0
    voiced = sum(_bp(glottal, f - 0.2 * f, f + 0.2 * f) * a for f, a in [(rng.uniform(300, 800), 1.0), (rng.uniform(900, 2200), 0.5)])
    voiced = voiced / (np.sqrt(np.mean(voiced ** 2)) + 1e-9)          # vowels are the loud part of speech (unit loudness)
    x = np.zeros(n)
    c = 0.1
    while c < dur - 0.4:
        L = int(rng.uniform(0.10, 0.28) * SR0)
        i = int(c * SR0)
        if i + L < n:
            x[i:i + L] += np.hanning(L) * voiced[i:i + L]
            B = int(rng.uniform(0.008, 0.03) * SR0)         # consonant burst at the start of the syllable
            if i - B > 0 and rng.random() < 0.6:
                x[i - B:i] += rng.uniform(0.15, 0.6) * _hp(_white(B, rng), 400) * np.hanning(B)   # consonants are weaker than vowels
        c += rng.uniform(0.18, 0.4)
        if rng.random() < 0.15:
            c += rng.uniform(0.3, 0.9)
    return x + 0.02 * _pink(n, rng)


def rustle(dur, rng):
    """leaves, fabric or paper: many tiny crackles, 40-150 per second"""
    n, _ = _t(dur)
    x = np.zeros(n)
    rate = rng.uniform(40, 150)
    for ti in rng.uniform(0, dur - 0.02, int(rate * dur)):
        i, L = int(ti * SR0), int(rng.uniform(0.002, 0.008) * SR0)
        x[i:i + L] += rng.uniform(0.1, 1.0) * _white(L, rng) * np.hanning(L)
    return x * _smooth_env(n, rng, 1.2, 0.7, floor=0.3) + 0.02 * _pink(n, rng)


def handling(dur, rng):
    """the phone or laptop being held and shifted: slow thumps and scratches"""
    n, _ = _t(dur)
    x = 0.04 * _pink(n, rng)
    for ti in rng.uniform(0, dur - 0.3, rng.integers(2, 5)):
        i, L = int(ti * SR0), int(rng.uniform(0.1, 0.3) * SR0)
        x[i:i + L] += rng.uniform(0.5, 1.2) * _bp(_white(L, rng), 60, 700) * np.hanning(L)
    return x


def traffic(dur, rng):
    n, _ = _t(dur)
    return _lp(_white(n, rng), 300, order=3) * _smooth_env(n, rng, 0.3, 0.6, floor=0.4) + 0.05 * _pink(n, rng)


def outdoor_mix(dur, rng):
    """outdoors: wind gusts, distant traffic and a little rustling together"""
    return 1.0 * air_gusts(dur, rng) + 0.6 * traffic(dur, rng) + 0.5 * rustle(dur, rng)


def babble(dur, rng):
    """a crowd talking: six made-up speakers on top of each other (an exhibition hall)"""
    return sum(speech_like(dur, rng) for _ in range(6))


def footsteps(dur, rng):
    """footsteps on a hard floor: a low thump with a little click, about 2 per second"""
    n, _ = _t(dur)
    x = 0.03 * _pink(n, rng)
    t = 0.2
    while t < dur - 0.2:
        i, L = int(t * SR0), int(0.12 * SR0)
        if i + L < n:
            x[i:i + L] += rng.uniform(0.6, 1.2) * _impulse(L, rng.uniform(60, 160), rng.uniform(25, 60), rng, 0.25)
            C = int(0.01 * SR0)
            x[i:i + C] += rng.uniform(0.1, 0.5) * _white(C, rng) * np.hanning(C)
        t += rng.uniform(0.4, 0.7)
    return x


def clatter(dur, rng):
    """a few dropped or knocked objects: 3-8 sharp impacts at random moments over a quiet room"""
    n, _ = _t(dur)
    x = 0.02 * _pink(n, rng)
    for ti in rng.uniform(0.3, dur - 0.5, rng.integers(3, 9)):
        i, L = int(ti * SR0), int(0.08 * SR0)
        x[i:i + L] += rng.uniform(0.5, 1.2) * _impulse(L, rng.uniform(250, 1200), rng.uniform(40, 150), rng, 0.4)
    return x


def hall(dur, rng):
    """exhibition hall: crowd babble, footsteps in the distance and now and then a clatter"""
    return 0.5 * babble(dur, rng) + 0.6 * footsteps(dur, rng) + 0.7 * clatter(dur, rng)


# ---------------------------------------------------------------- short real pulses that are not larvae (the honest limit)

def keyboard_clicks(dur, rng):
    n, _ = _t(dur)
    x = 0.02 * _pink(n, rng)
    t = 0.3
    while t < dur - 0.1:
        for _ in range(rng.integers(1, 4)):
            i, L = int(t * SR0), int(0.012 * SR0)
            if i + L < n:
                x[i:i + L] += rng.uniform(0.4, 1.0) * _white(L, rng) * np.exp(-np.arange(L) / (0.003 * SR0))
            t += rng.uniform(0.07, 0.18)
        t += rng.uniform(0.2, 0.9)
    return x


def finger_taps(dur, rng):
    """tapping a table or a trunk with a fingernail, 1-3 taps per second"""
    n, _ = _t(dur)
    x = 0.02 * _pink(n, rng)
    t = 0.3
    while t < dur - 0.1:
        i, L = int(t * SR0), int(0.05 * SR0)
        if i + L < n:
            x[i:i + L] += rng.uniform(0.6, 1.0) * _impulse(L, rng.uniform(300, 900), rng.uniform(60, 120), rng, 0.3)
        t += rng.uniform(0.3, 0.9)
    return x


# ---------------------------------------------------------------- weevil-like (what must still be heard)

def _weevil_pulse(L, freq, decay):
    t = np.arange(L) / SR0
    env = np.exp(-t * decay)
    return env * np.sin(2 * np.pi * freq * t) + 0.35 * env * np.sin(2 * np.pi * 2.1 * freq * t)


def weevil(dur, rng, rate=1.5, amp=0.6, bg=0.12, bg_kind="pink"):
    """simulated chewing: bursts of 2-4 damped 400-560 Hz pulses on a background (same recipe as make_weevil_sound.py)"""
    n, _ = _t(dur)
    back = {"pink": _pink, "white": _white, "gust": lambda n_, r: air_gusts(n_ / SR0, r)}[bg_kind](n, rng)
    back = back / (np.max(np.abs(back)) + 1e-9)
    x = bg * back
    t = 0.3
    while t < dur - 0.2:
        k = rng.integers(2, 5)
        for j in range(k):
            i, L = int((t + j * rng.uniform(0.07, 0.16)) * SR0), int(0.06 * SR0)
            if i + L < n:
                x[i:i + L] += amp * rng.uniform(0.75, 1.25) * _weevil_pulse(L, rng.uniform(400, 560), 55)
        t += k / rate * rng.uniform(0.8, 1.2)
    return x


# ---------------------------------------------------------------- the judges' demo path: a sound played from a phone speaker, recorded by a laptop microphone

def via_speaker(x, rng, rt60=None, hp=300.0, room_snr_db=None, drr_db=None):
    """phone speaker -> small room -> laptop microphone placed 5-10 cm away: no sound under ~300 Hz, a reverberant tail
    (RT60 0.15-0.6 s) that is 6-15 dB weaker than the direct sound, room noise on top"""
    from scipy.signal import fftconvolve
    rt60 = rng.uniform(0.15, 0.6) if rt60 is None else rt60
    drr = rng.uniform(6, 15) if drr_db is None else drr_db
    L = int(rt60 * SR0 * 1.2)
    t = np.arange(L) / SR0
    tail = rng.normal(size=L) * np.exp(-t * 6.9 / rt60)
    tail = tail / np.sqrt(np.sum(tail ** 2)) * 10 ** (-drr / 20)       # tail energy = direct energy / 10^(DRR/10)
    rir = tail
    rir[0] += 1.0                                                       # direct sound
    y = fftconvolve(_hp(x, hp), rir)[: len(x)]
    y = y / (np.max(np.abs(y)) + 1e-9)
    snr = rng.uniform(18, 35) if room_snr_db is None else room_snr_db
    noise = _pink(len(y), rng)
    noise = noise / (np.sqrt(np.mean(noise ** 2)) + 1e-9) * np.sqrt(np.mean(y ** 2)) * 10 ** (-snr / 20)
    return y + noise


def speaker_weevil_dense(dur, rng):
    return via_speaker(weevil(dur, rng, rate=2.6, amp=0.9, bg=0.12), rng)


def speaker_weevil_sparse(dur, rng):
    return via_speaker(weevil(dur, rng, rate=0.6, amp=0.6, bg=0.12), rng)


def speaker_weevil_far(dur, rng):
    """stress case: the phone is held further away (direct sound only 0-4 dB above the reverberation) in a lively room"""
    return via_speaker(weevil(dur, rng, rate=2.0, amp=0.8, bg=0.12), rng, rt60=rng.uniform(0.4, 0.7), drr_db=rng.uniform(0, 4))


def speaker_air(dur, rng):
    """the 'air only' demo sound (low-passed, gusty noise) played from the phone"""
    n, t = _t(dur)
    low = _lp(_white(n, rng), 900)
    gust = 0.6 + 0.4 * np.sin(2 * np.pi * 0.35 * t + rng.uniform(0, 6))
    return via_speaker(0.8 * low * gust + 0.25 * _pink(n, rng), rng)


def speaker_babble(dur, rng):
    return via_speaker(babble(dur, rng), rng)


NOT_WEEVIL = {
    "white_room": white_room, "pink_room": pink_room, "mains_hum": mains_hum, "fan": fan, "air_gusts": air_gusts,
    "air_on_mic": air_on_mic, "mic_pops": mic_pops, "breathing": breathing, "blowing": blowing, "blowing_loud": blowing_loud, "speech_like": speech_like,
    "rustle": rustle, "handling": handling, "traffic": traffic, "outdoor_mix": outdoor_mix, "babble": babble, "footsteps": footsteps,
}
SHORT_NOT_LARVAE = {"keyboard_clicks": keyboard_clicks, "finger_taps": finger_taps, "clatter": clatter, "hall": hall}
WEEVIL = {
    "weevil_dense": lambda d, r: weevil(d, r, rate=2.6, amp=0.9, bg=0.12),
    "weevil_sparse": lambda d, r: weevil(d, r, rate=0.6, amp=0.6, bg=0.12),
    "weevil_weak_pink": lambda d, r: weevil(d, r, rate=1.5, amp=0.35, bg=0.2),
    "weevil_on_white": lambda d, r: weevil(d, r, rate=1.5, amp=0.5, bg=0.25, bg_kind="white"),
    "weevil_on_gusts": lambda d, r: weevil(d, r, rate=1.5, amp=0.7, bg=0.5, bg_kind="gust"),
}

SPEAKER_NOT_WEEVIL = {"speaker_air": speaker_air, "speaker_babble": speaker_babble}
SPEAKER_WEEVIL = {"speaker_weevil_dense": speaker_weevil_dense, "speaker_weevil_sparse": speaker_weevil_sparse, "speaker_weevil_far": speaker_weevil_far}
