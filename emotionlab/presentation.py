"""Format hasil nyata, tanpa generator bahasa eksternal."""
import csv
import json
from pathlib import Path
from .comparison import export as export_signal


def explain(result):
    if 'message' in result:return result['message']
    if result['kind']=='metadata':
        lines=[]
        for key,info in result['records'].items():
            lines += [f"Rekaman {key}: {Path(info['path']).name}",f"Durasi: {info['duration_s']:.1f} detik; kanal: {len(info['channels'])}; anotasi: {len(info['annotations'])}.",
                      'Kanal: '+', '.join(c['name'] for c in info['channels']),
                      'Sampling: '+', '.join(f'{fs:g} Hz' for fs in sorted({c['sample_rate'] for c in info['channels']})),'']
        return '\n'.join(lines)
    if result['kind']=='signal':
        report=result['result']['report']
        return f"Perbandingan sinyal {len(report['channels'])} kanal, durasi {report['duration_s']:g} detik.\n"+'\n'.join(report['warnings'])
    lines=['HASIL PREDIKSI MODEL EEG — bukan label bawaan rekaman',
           'Eksperimen awal dari dua subjek. Kinerja uji masih rendah; hasil belum dapat diandalkan untuk menilai emosi seseorang.']
    for key,timeline in result['timelines'].items():
        lines+=['',f"Rekaman {key} | {Path(timeline['file']).name} | model {timeline['model']}",
                f"{len(timeline['rows'])} window × {timeline['window_s']:g} detik. Skor adalah keluaran model, belum dikalibrasi.",
                f"Akurasi model pada subjek uji: {timeline['evaluation']['accuracy']:.1%}; macro-F1: {timeline['evaluation']['macro_f1']:.3f}."]
        if timeline['training_subject_overlap']:lines.append('Rekaman berasal dari subjek latihan; hasil ini bukan evaluasi independen.')
        if timeline['ignored_tail_s']>.001:lines.append(f"Sisa {timeline['ignored_tail_s']:.2f} detik tidak cukup untuk satu window dan tidak diprediksi.")
        summary=result['summaries'][key]
        lines.append('Durasi prediksi: '+', '.join(f"{label} {v['seconds']:g}s ({v['percent']:.1f}%)" for label,v in summary.items() if v['seconds']>0))
        if result['kind']=='find':lines.append(f"Ditemukan {len(result['matches'][key])} window untuk emosi yang dicari. 'Belum yakin' tidak dianggap kecocokan.")
    if result['kind']=='compare_emotion':
        lines+=['','Perbandingan proporsi dalam rentang pilihan; bukan kesamaan waktu eksperimen atau uji signifikansi.']
        lines += [f"{r['emotion']}: A {r['A_percent']:.1f}%, B {r['B_percent']:.1f}%, selisih {r['B_minus_A_pp']:+.1f} poin persentase." for r in result['comparison']]
    return '\n'.join(lines)


def timeline_rows(result):
    rows=[]
    for key,timeline in result.get('timelines',{}).items():
        source=result['matches'][key] if result['kind']=='find' else timeline['rows']
        rows.extend({'record':key,**row} for row in source)
    return rows


def export_result(result,folder):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    if result['kind']=='signal':return export_signal(result['result'],folder)
    (folder/'hasil.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    (folder/'penjelasan.txt').write_text(explain(result),encoding='utf-8')
    rows=timeline_rows(result)
    if rows:
        columns=['record','start_s','end_s','prediction','display','score','margin','distribution_shift']
        with (folder/'timeline.csv').open('w',newline='',encoding='utf-8-sig') as f:
            writer=csv.DictWriter(f,fieldnames=columns,extrasaction='ignore');writer.writeheader();writer.writerows(rows)
    return folder
