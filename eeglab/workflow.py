"""Aturan alur dan penjelasan untuk pengguna, tanpa dependensi GUI."""
import json
from pathlib import Path
from .edf_reader import inspect_edf,common_voltage_channels
from .events import read_events
from .training import load_manifest,group_split,resolve_training_channels

METRICS={
 'mean_uv':('Rata-rata amplitudo','µV'), 'std_uv':('Variasi amplitudo (std)','µV'),
 'rms_uv':('Amplitudo efektif (RMS)','µV'), 'peak_to_peak_uv':('Rentang puncak ke puncak','µV'),
 'power_delta_uv2':('Daya delta · 1–4 Hz','µV²'), 'power_theta_uv2':('Daya theta · 4–8 Hz','µV²'),
 'power_alpha_uv2':('Daya alpha · 8–13 Hz','µV²'), 'power_beta_uv2':('Daya beta · 13–30 Hz','µV²'),
 'power_gamma_uv2':('Daya gamma · 30–45 Hz','µV²')}


def valid_edf(path):
    if not str(path).strip():raise ValueError('Pilih file EDF untuk kedua rekaman dahulu. File yang sama boleh dipilih dua kali.')
    p=Path(path).expanduser().resolve()
    if not p.is_file() or p.suffix.lower()!='.edf':raise ValueError('Pilih file .edf yang sudah selesai diunduh: '+str(p))
    return str(p)


def suggested_sidecars(path):
    p=Path(path);stem=p.stem
    json_path=p.with_suffix('.json')
    beh_stem=stem[:-4]+'_beh.tsv' if stem.endswith('_eeg') else stem+'_beh.tsv'
    event_stem=stem[:-4]+'_events.tsv' if stem.endswith('_eeg') else stem+'_events.tsv'
    candidates=[p.parent/event_stem,p.parent/beh_stem,p.parent.parent/'beh'/beh_stem]
    return {'json':str(json_path) if json_path.is_file() else '',
            'tsv':next((str(c) for c in candidates if c.is_file()),'')}


def inspect_pair(paths,sidecars):
    result={'info':{},'sidecars':{},'warnings':[]}
    for key in ('A','B'):
        result['info'][key]=inspect_edf(valid_edf(paths[key]))
        extras={}
        for kind,path in sidecars[key].items():
            if not path.strip():continue
            try:
                if kind=='json':
                    with open(path,encoding='utf-8-sig') as file:data=json.load(file)
                    if not isinstance(data,dict):raise ValueError('JSON harus objek metadata.')
                    extras['json']={'path':path,'data':data}
                else:extras['tsv']=read_events(path)
            except Exception as exc:result['warnings'].append(f'Pendamping {key} ({kind}) tidak dibaca: {exc}')
        result['sidecars'][key]=extras
    result['channels']=common_voltage_channels(result['info'].values())
    if not result['channels']:raise ValueError('Tidak ada kanal tegangan bernama sama pada kedua EDF. Periksa montase/nama kanal.')
    return result


def manifest_overview(path):
    rows=load_manifest(path)
    channels=resolve_training_channels(rows,'auto')
    overview={'rows':len(rows),'files':len({r['file'] for r in rows}),
              'groups':len({r['group'] for r in rows}),'labels':sorted({r['label'] for r in rows}),
              'channels':channels,'ready':True,'reason':''}
    try:group_split(rows)
    except ValueError as exc:overview.update(ready=False,reason=str(exc))
    return overview


def channel_summary(report,channel):
    rows=[r for r in report['comparison'] if r['channel']==channel]
    lines=[f'Kanal {channel} · {report["duration_s"]:g} detik',
           f"A: mulai detik {report['start_A_s']:g}  |  B: mulai detik {report['start_B_s']:g}",
           '', 'Apa yang berbeda?']
    for row in rows:
        if row['metric'] not in ('rms_uv','power_alpha_uv2','power_beta_uv2') or row['A'] is None:continue
        name,unit=METRICS[row['metric']];delta=row['B_minus_A']
        direction='lebih tinggi' if delta>0 else 'lebih rendah' if delta<0 else 'sama'
        lines.append(f"• {name}: A {row['A']:.4g} {unit}; B {row['B']:.4g} {unit}. B {direction} ({delta:+.4g} {unit}).")
    lines+=['','Cara membaca hasil:',
            'RMS menggambarkan amplitudo efektif. Daya pita menggambarkan energi kuadrat sinyal pada rentang frekuensi tertentu.',
            'Selisih B − A positif berarti nilai B lebih besar. Ini perbandingan deskriptif, belum uji beda statistik.',
            '', 'Hal yang perlu diperhatikan:']+['• '+w for w in report['warnings']]
    return '\n'.join(lines)
