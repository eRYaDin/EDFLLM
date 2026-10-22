"""Prediksi emosi per window; label dataset tidak pernah menjadi input prediktor."""
import json
from pathlib import Path
import numpy as np
from .dataset import ROOT,features
from .training import load_model
from .signal import Settings,prepare
from .edf import inspect_edf,common_voltage_channels


def resolve_model(choice='auto',folder=ROOT/'models/emotion'):
    folder=Path(folder);report_path=folder/'training_report.json'
    if not report_path.is_file():raise ValueError('Model EEG belum dilatih. Buka tab Persiapan lalu Latih tujuh jaringan.')
    report=json.loads(report_path.read_text(encoding='utf-8'))
    if choice=='auto':model_id=report['selected']
    else:
        candidates=[r for r in report['results'] if r['network']==choice]
        if not candidates:raise ValueError('Pilihan jaringan tidak tersedia.')
        model_id=max(candidates,key=lambda r:r['validation']['macro_f1'])['id']
    return folder/(model_id+'.npz')


def predict_timeline(edf,start=0.,duration=28.,choice='auto',model_dir=ROOT/'models/emotion',log=print):
    info=inspect_edf(edf);path=resolve_model(choice,model_dir);net,meta,mean,std=load_model(path)
    settings=Settings(**meta['settings']);start=float(start);duration=float(duration)
    if not np.isfinite([start,duration]).all() or start<0 or duration<settings.window_s or duration>info['duration_s']-start+1e-6:
        raise ValueError('Segmen harus berada di dalam rekaman dan minimal satu window (4 detik).')
    missing=set(meta['channels'])-set(common_voltage_channels([info]))
    if missing:raise ValueError('EDF tidak sesuai kanal model: '+', '.join(sorted(missing)))
    # Chunking is bounded in memory and aligns windows across chunks.
    count=int(np.floor((duration+1e-6)/settings.window_s));rows=[]
    for first in range(0,count,15):
        n=min(15,count-first);begin=start+first*settings.window_s
        log(f'Prediksi window {first+1}–{first+n} / {count}')
        signal,quality=prepare(edf,meta['channels'],begin,n*settings.window_s,settings)
        x,names=features(signal,meta['channels'],settings)
        if names!=meta['features']:raise ValueError('Skema fitur tidak cocok.')
        z=(x-mean)/std;scores=net.predict_scores(z)
        for i,score in enumerate(scores):
            order=np.argsort(score);top=int(order[-1]);confidence=float(score[top]);margin=float(score[order[-1]]-score[order[-2]])
            distribution_shift=float(np.mean(np.abs(z[i])>6))>.1
            certain=confidence>=.45 and margin>=.08 and not distribution_shift
            rows.append({'start_s':begin+i*settings.window_s,'end_s':begin+(i+1)*settings.window_s,
                         'prediction':meta['labels'][top],'display':meta['labels'][top] if certain else 'belum yakin',
                         'score':confidence,'margin':margin,'distribution_shift':distribution_shift,
                         'scores':dict(zip(meta['labels'],map(float,score)))})
    return {'file':str(Path(edf).resolve()),'model':meta['id'],'window_s':settings.window_s,'rows':rows,
            'labels':meta['labels'],'label_source':meta['label_source'],'ignored_tail_s':max(0.,duration-count*settings.window_s),
            'note':'Prediksi kategori emosi stimulus; skor bukan kepastian perasaan. Ambang belum dikalibrasi.',
            'evaluation':meta['test'],'training_subject_overlap':any(subject in Path(edf).name for subject in meta['train_subjects'])}


def summarize_timeline(result):
    duration={label:0. for label in [*result['labels'],'belum yakin']}
    for row in result['rows']:duration[row['display']]+=row['end_s']-row['start_s']
    total=sum(duration.values())
    return {label:{'seconds':seconds,'percent':100*seconds/total if total else 0.} for label,seconds in duration.items()}


def compare_emotions(a,b):
    if a['model']!=b['model'] or a['labels']!=b['labels']:raise ValueError('Perbandingan harus memakai model dan label yang sama.')
    sa,sb=summarize_timeline(a),summarize_timeline(b)
    return [{'emotion':label,'A_percent':sa[label]['percent'],'B_percent':sb[label]['percent'],
             'B_minus_A_pp':sb[label]['percent']-sa[label]['percent']} for label in sa]
