from pathlib import Path
from datetime import datetime
import json
import numpy as np
from .edf import inspect_edf
from .signal import prepare
from .features import describe


def compare(a, b, channels, start_a, start_b, duration, settings, reference_a='', reference_b=''):
    if not 4 <= duration <= 600:
        raise ValueError('Durasi perbandingan 4–600 detik.')
    info_a, info_b = inspect_edf(a), inspect_edf(b)
    x, qa = prepare(a, channels, start_a, duration, settings)
    y, qb = prepare(b, channels, start_b, duration, settings)
    warnings = ['Perbedaan deskriptif bukan diagnosis atau bukti perbedaan emosi.',
                'Segmen dimulai pada waktu pilihan pengguna; belum diselaraskan otomatis berdasarkan stimulus.',
                'Filter zero-phase memakai konteks tambahan hingga 2 detik bila tersedia; tepi file masih dapat bertransien. Tidak ada ICA otomatis.']
    if not reference_a.strip() or not reference_b.strip() or reference_a != reference_b:
        warnings.append('Referensi elektroda belum dinyatakan sama. Interpretasi amplitudo/power perlu ditinjau.')
    if info_a['start_time'] != info_b['start_time']:
        warnings.append('Waktu mulai rekaman berbeda; sumbu grafik menunjukkan waktu relatif segmen.')
    table = []
    spectra = {}
    for i, channel in enumerate(channels):
        da, f, pa = describe(x[i], settings)
        db, _, pb = describe(y[i], settings)
        spectra[channel] = (f, pa, pb)
        for key in da:
            va, vb = da[key], db[key]
            table.append({'channel':channel, 'metric':key, 'A':va, 'B':vb,
                          'B_minus_A': vb-va if va is not None and vb is not None else None})
    report = {'created':datetime.now().astimezone().isoformat(), 'file_A':info_a, 'file_B':info_b,
              'channels':list(channels), 'start_A_s':start_a, 'start_B_s':start_b, 'duration_s':duration,
              'settings':settings.to_dict(), 'reference_A':reference_a, 'reference_B':reference_b,
              'quality_A':qa, 'quality_B':qb, 'warnings':warnings, 'comparison':table,
              'method':'statistik dan PSD, tanpa prediksi ML'}
    return {'report':report, 'signals_A':x, 'signals_B':y, 'spectra':spectra}


def export(result, folder):
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    import csv
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    report = result['report']
    (folder/'laporan.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    with (folder/'perbandingan.csv').open('w',newline='',encoding='utf-8-sig') as file:
        writer=csv.DictWriter(file,fieldnames=['channel','metric','A','B','B_minus_A'])
        writer.writeheader(); writer.writerows(report['comparison'])
    channel=report['channels'][0]
    fig=Figure(figsize=(10,7),tight_layout=True); FigureCanvasAgg(fig)
    time=np.arange(result['signals_A'].shape[1])/report['settings']['fs']
    ax=fig.add_subplot(211)
    ax.plot(time,result['signals_A'][0],label='A',alpha=.8,lw=.7)
    ax.plot(time,result['signals_B'][0],label='B',alpha=.7,lw=.7)
    ax.set(xlabel='Waktu relatif (s)',ylabel='Amplitudo (uV)',title=channel+' | segmen terfilter');ax.legend();ax.grid(alpha=.2)
    ax=fig.add_subplot(212);f,pa,pb=result['spectra'][channel]
    ax.semilogy(f,np.maximum(pa,1e-20),label='A');ax.semilogy(f,np.maximum(pb,1e-20),label='B')
    ax.set(xlabel='Frekuensi (Hz)',ylabel='PSD (uV^2/Hz)',xlim=(0,report['settings']['fs']/2));ax.legend();ax.grid(alpha=.2)
    fig.savefig(folder/'grafik.png',dpi=140)
    return folder
