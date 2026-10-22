import numpy as np
from scipy.signal import welch

BANDS = {'delta': (1.,4.), 'theta': (4.,8.), 'alpha': (8.,13.),
         'beta': (13.,30.), 'gamma': (30.,45.)}


def spectrum(x, fs):
    return welch(x, fs=fs, window='hann', nperseg=min(len(x), round(2*fs)),
                 detrend='constant', scaling='density')


def bandpower(freq, psd, low, high):
    inside = (freq > low) & (freq < high)
    f = np.r_[low, freq[inside], high]
    p = np.interp(f, freq, psd)
    return float(np.trapezoid(p, f))


def describe(x, settings):
    freq, psd = spectrum(x, settings.fs)
    result = {'mean_uv': float(np.mean(x)), 'std_uv': float(np.std(x)),
              'rms_uv': float(np.sqrt(np.mean(x*x))), 'peak_to_peak_uv': float(np.ptp(x))}
    for name, (lo, hi) in BANDS.items():
        result['power_' + name + '_uv2'] = (bandpower(freq, psd, lo, hi)
             if settings.low <= lo and hi <= settings.high else None)
    return result, freq, psd


def window_features(signals, channels, settings):
    size = round(settings.window_s * settings.fs)
    names, rows = [], []
    if signals.shape[1] < size:
        raise ValueError('Segmen lebih pendek dari satu window.')
    for start in range(0, signals.shape[1] - size + 1, size):
        row, row_names = [], []
        for channel, x in zip(channels, signals[:,start:start+size]):
            values, _, _ = describe(x, settings)
            for key, value in values.items():
                if value is not None:
                    row_names.append(channel + ':' + key)
                    row.append(value)
        names = row_names
        rows.append(row)
    return np.asarray(rows), names
