from dataclasses import dataclass, asdict
from fractions import Fraction
import numpy as np
from scipy.signal import butter, sosfiltfilt, resample_poly
from .edf import read_segment,inspect_edf


@dataclass(frozen=True)
class Settings:
    fs: float = 128.
    low: float = 1.
    high: float = 45.
    window_s: float = 2.

    def validate(self):
        if not all(np.isfinite(x) for x in (self.fs, self.low, self.high, self.window_s)):
            raise ValueError('Parameter harus angka hingga.')
        if not 0 < self.low < self.high < self.fs / 2 or self.window_s < 2:
            raise ValueError('Perlu 0 < low < high < sampling/2 dan window minimal 2 detik.')
        if not 32 <= self.fs <= 2048 or self.window_s > 60:
            raise ValueError('Sampling 32–2048 Hz; window maksimal 60 detik.')

    def to_dict(self):
        return asdict(self)


def prepare(path, channels, start, duration, settings):
    settings.validate()
    if not np.isfinite([start,duration]).all() or start<0 or duration<2:
        raise ValueError('Awal >=0 dan durasi minimal 2 detik, berupa angka hingga.')
    file_duration=inspect_edf(path)['duration_s']
    if start+duration>file_duration+1e-6:raise ValueError('Segmen melewati akhir rekaman.')
    # Filter mendapat konteks dua detik jika tersedia, baru hasil dipotong ke segmen.
    left=max(0.,start-2.);right=min(file_duration,start+duration+2.)
    raw = read_segment(path, channels, left, right-left)
    signals, quality = [], {}
    target_n = round(duration * settings.fs)
    sos = butter(4, [settings.low, settings.high], btype='bandpass', fs=settings.fs, output='sos')
    for name in channels:
        item = raw[name]
        fs, x = item['native_fs'], item['values_uv']
        if settings.fs > fs + 1e-6:
            raise ValueError(f'{name}: target sampling melebihi sampling asli {fs:g} Hz. Turunkan target sampling.')
        segment_start=round((start-left)*fs)
        original=x[segment_start:segment_start+round(duration*fs)]
        quality[name] = {'raw_rms_uv': float(np.sqrt(np.mean(original*original))),
                         'raw_range_uv': float(np.ptp(original)),
                         'constant_step_fraction': float(np.mean(np.diff(original) == 0)),
                         'native_fs': fs, 'offset_rounding_s': item['offset_rounding_s'],
                         'filter_context_before_s':start-left,'filter_context_after_s':right-start-duration}
        ratio = Fraction(settings.fs / fs).limit_denominator(10000)
        x = resample_poly(x, ratio.numerator, ratio.denominator)
        x = sosfiltfilt(sos, x)
        cut=round((start-left)*settings.fs)
        x=x[cut:cut+target_n]
        if len(x) != target_n:
            raise ValueError('Panjang hasil resampling tidak sesuai durasi yang diminta.')
        signals.append(x)
    return np.asarray(signals), quality
