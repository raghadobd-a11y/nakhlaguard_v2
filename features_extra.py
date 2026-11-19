"""
Extra features aimed at the signature of feeding larvae: short, sharp, repeated pulses on top of background noise.   (ASCII-only)
The 44 base features describe the average spectrum. These 14 add:
  - how impulsive the signal is (crest factor, kurtosis, skewness, envelope variation)
  - whether the pulses repeat (envelope autocorrelation peak and its lag, pulse rate, regularity of the gaps between pulses)
  - how noise-like the spectrum is (flatness, 85% roll-off) and how the energy is split between low, middle and high bands.
Steady wind or engine noise is not impulsive and not periodic; a feeding insect is.
5 more features follow the burst definition of Mankin et al. 2016 (J. Econ. Entomol. 109: 622-628): impulses are grouped into trains when
the gap is under 25 ms; a train of at least 7 and fewer than 200 impulses is an "insect sound burst"; longer trains are mostly wind. Whether these features really help on
real recordings must be MEASURED (see: python train_treevibes.py cv ...), not assumed.
"""
import numpy as np
from scipy.signal import hilbert, find_peaks, fftconvolve
from features import extract_features, FEATURE_NAMES

EXTRA_NAMES = ["crest_factor", "kurtosis", "skewness", "envelope_cv", "env_autocorr_peak", "env_autocorr_lag_s", "pulse_rate_per_s",
               "pulse_interval_cv", "spectral_flatness", "spectral_rolloff_85", "band_20_200_share", "band_200_800_share", "band_800_plus_share", "log_rms",
               "burst_rate_per_s", "impulses_per_burst", "burst_impulse_rate_per_s", "long_train_fraction", "impulse_rate_per_s"]
FEATURE_NAMES_EXTRA = list(FEATURE_NAMES) + EXTRA_NAMES


def burst_features(x, sr):
    """Impulses = peaks of the 1 ms-smoothed envelope far above the robust noise level (median + 5 MAD), at least 3 ms apart."""
    env = np.abs(hilbert(x)); k = max(1, int(0.001 * sr)); env = np.convolve(env, np.ones(k) / k, mode="same")
    med = np.median(env); mad = 1.4826 * np.median(np.abs(env - med)) + 1e-12
    peaks, _ = find_peaks(env, height=med + 5 * mad, distance=max(1, int(0.003 * sr)))
    dur = len(x) / sr; t = peaks / sr
    if len(t) == 0:
        return [0.0, 0.0, 0.0, 0.0, 0.0]
    trains = np.split(t, np.where(np.diff(t) >= 0.025)[0] + 1); sizes = np.array([len(tr) for tr in trains])
    bursts = sizes[(sizes >= 7) & (sizes < 200)]; long_n = sizes[sizes >= 200].sum()
    return [len(bursts) / dur, float(bursts.mean()) if len(bursts) else 0.0, float(bursts.sum()) / dur, float(long_n) / float(sizes.sum()), len(t) / dur]


def extra_features(audio, sr=4000):
    x = np.asarray(audio, dtype=np.float64)
    x = x - x.mean(); eps = 1e-12
    sd = x.std() + eps; rms = np.sqrt(np.mean(x ** 2)) + eps
    crest = np.max(np.abs(x)) / rms
    z = x / sd; kurt = np.mean(z ** 4); skew = np.mean(z ** 3)
    env = np.abs(hilbert(x)); k = max(1, int(0.01 * sr))
    envs = np.convolve(env, np.ones(k) / k, mode="same"); envs = envs / (envs.mean() + eps)
    env_cv = envs.std()
    ds = max(1, sr // 400); e = envs[::ds]; fs = sr / ds; e0 = e - e.mean()                      # envelope at about 400 Hz
    ac = fftconvolve(e0, e0[::-1], mode="full")[len(e0) - 1:]; ac = ac / (ac[0] + eps)
    lo, hi = max(1, int(0.02 * fs)), min(len(ac) - 1, int(0.4 * fs))
    seg = ac[lo:hi] if hi > lo else np.array([0.0])
    ac_peak = float(seg.max()); ac_lag = float((lo + int(np.argmax(seg))) / fs)
    peaks, _ = find_peaks(e, height=e.mean() + 1.5 * e.std(), distance=max(1, int(0.03 * fs)))
    rate = len(peaks) / (len(x) / sr)
    gaps = np.diff(peaks)
    gap_cv = float(np.std(gaps) / (np.mean(gaps) + eps)) if len(peaks) > 3 else 3.0
    sp = np.abs(np.fft.rfft(x * np.hanning(len(x)))) ** 2 + eps; f = np.fft.rfftfreq(len(x), 1 / sr)
    flat = float(np.exp(np.mean(np.log(sp))) / np.mean(sp))
    cs = np.cumsum(sp); roll = float(f[min(len(f) - 1, int(np.searchsorted(cs, 0.85 * cs[-1])))])
    low = sp[(f >= 20) & (f < 200)].sum(); mid = sp[(f >= 200) & (f < 800)].sum(); high = sp[f >= 800].sum(); tot = low + mid + high + eps
    ex = np.array([crest, kurt, skew, env_cv, ac_peak, ac_lag, rate, gap_cv, flat, roll, low / tot, mid / tot, high / tot, np.log(rms)] + burst_features(x, sr), dtype=np.float64)
    ex = np.nan_to_num(ex, nan=0.0, posinf=1e6, neginf=-1e6)
    return np.concatenate([extract_features(audio, sr), ex]).astype(np.float32)
