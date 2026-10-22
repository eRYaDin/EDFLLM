"""Adapter EmoEEG-MC sub-02/03: label stimulus, bukan pembacaan perasaan aktual.

Sumber pemetaan: README dataset ds005540, bagian Full Trial Participants dan
Map Triggers to Trial Types. Urutan stimulus dihitung per context, bukan urutan
baris beh.tsv. Jangan gunakan adapter ini untuk subjek lain tanpa profilnya.
"""
import csv
import hashlib
import json
import re
from pathlib import Path
import numpy as np
from scipy.signal import welch
from .edf import inspect_edf,common_voltage_channels
from .signal import prepare,Settings

ROOT=Path(__file__).resolve().parent.parent
LABELS=['senang','terinspirasi','kelembutan','takut','jijik','sedih','netral']
PREFIX={'joy':'senang','ins':'terinspirasi','ten':'kelembutan','fear':'takut','dis':'jijik','sad':'sedih','neu':'netral'}
SEQUENCES={
 'sub-02':'fear8 fear5 dis4 ins8 joy8 ins4 neu4 neu8 neu5 sad4 dis8 fear4 ten5 ten8 joy4 dis5 sad8 sad5 joy5 ten4 ins5'.split(),
 'sub-03':'ten4 joy4 joy8 neu4 neu8 neu5 dis5 fear4 fear5 ten8 ten5 ins5 fear8 dis4 dis8 ins8 joy5 ins4 sad4 sad5 sad8'.split()}
SOURCE='https://github.com/OpenNeuroDatasets/ds005540'
BANDS={'delta':(1,4),'theta':(4,8),'alpha':(8,13),'beta':(13,30),'gamma':(30,45)}
SETTINGS=Settings(window_s=4.)


def trials_from_info(info,subject):
    if subject not in SEQUENCES:raise ValueError('Profil urutan stimulus belum tersedia untuk '+subject)
    events=[]
    for a in info['annotations']:
        match=re.fullmatch(r'TypeID:\s*(\d+)',a['description'].strip())
        if match:events.append((a['onset_s'],int(match[1])))
    events.sort();counts={32:0,33:0};rows=[]
    if any(sum(code==wanted for _,code in events)!=21 for wanted in counts):
        raise ValueError(subject+': perlu tepat 21 trigger imagery dan 21 video; pemetaan dihentikan agar label tidak bergeser.')
    for i,(onset,code) in enumerate(events):
        if code not in counts:continue
        context='imagery' if code==32 else 'video'
        stimulus=SEQUENCES[subject][counts[code]];counts[code]+=1
        following=events[i+1:]
        # Stop before fade/rating or next stimulus, whichever comes first.
        boundary=next((t for t,c in following if c in (30,31,32,33)),None)
        if boundary is None or not onset<boundary<=info['duration_s']:
            raise ValueError('Batas trial tidak ditemukan: '+stimulus)
        # Conservative project policy: skip initial second, use at most 28 s,
        # leave one second before boundary. No joining across fades/rest/rating.
        start=onset+1.;duration=min(28.,boundary-1.-start)
        if duration<4:raise ValueError('Trial terlalu pendek untuk window: '+stimulus)
        label=PREFIX[re.sub(r'\d+$','',stimulus)]
        rows.append({'file':info['path'],'subject':subject,'context':context,
                     'trial_id':f'{subject}:{context}:{counts[code]:02d}',
                     'stimulus':stimulus,'label':label,'onset_s':onset,
                     'boundary_s':boundary,'start_s':start,'duration_s':duration,
                     'label_source':'stimulus category from dataset README',
                     'split':('test' if subject=='sub-03' else 'validation' if stimulus.endswith('8') else 'train')})
    return rows


def build_manifest(raw_dir=ROOT/'data/raw'):
    rows=[];infos=[]
    for subject in SEQUENCES:
        path=Path(raw_dir)/(subject+'_task-emotion_eeg.edf')
        if not path.is_file():raise ValueError('EDF belum tersedia: '+str(path))
        info=inspect_edf(path);infos.append(info);rows.extend(trials_from_info(info,subject))
    channels=common_voltage_channels(infos)
    if not channels:raise ValueError('Tidak ada kanal tegangan bersama.')
    return rows,channels


def features(signals,channels,settings=SETTINGS):
    """Log band power and log RMS; no event labels/subject/time as predictors."""
    size=round(settings.window_s*settings.fs);n=signals.shape[1]//size
    if n<1:raise ValueError('Segmen lebih pendek dari satu window.')
    windows=signals[:,:n*size].reshape(len(channels),n,size).transpose(1,0,2)
    f,p=welch(windows,fs=settings.fs,nperseg=min(size,round(2*settings.fs)),axis=-1)
    columns=[];names=[]
    for ch,name in enumerate(channels):
        for band,(lo,hi) in BANDS.items():
            mask=(f>=lo)&(f<=hi)
            power=np.trapezoid(p[:,ch,mask],f[mask],axis=-1)
            columns.append(np.log10(np.maximum(power,1e-12)));names.append(name+':log_power_'+band)
        columns.append(np.log10(np.maximum(np.sqrt(np.mean(windows[:,ch]**2,axis=-1)),1e-12)))
        names.append(name+':log_rms')
    x=np.stack(columns,axis=1)
    if not np.isfinite(x).all():raise ValueError('Fitur tidak hingga.')
    return x,names


def prepare_dataset(raw_dir=ROOT/'data/raw',cache_dir=ROOT/'output/dataset',log=print):
    rows,channels=build_manifest(raw_dir);cache_dir=Path(cache_dir);cache_dir.mkdir(parents=True,exist_ok=True)
    stamp={'schema':1,'channels':channels,'settings':SETTINGS.to_dict(),'rows':rows,
           'files':{r['file']:[Path(r['file']).stat().st_size,Path(r['file']).stat().st_mtime_ns] for r in rows}}
    fingerprint=hashlib.sha256(json.dumps(stamp,sort_keys=True).encode()).hexdigest()
    cache=cache_dir/'features.npz'
    if cache.is_file():
        with np.load(cache,allow_pickle=False) as p:
            if str(p['fingerprint'])==fingerprint:
                return {k:p[k].copy() for k in p.files}
    xs=[];ys=[];splits=[];window_rows=[];names=None
    for index,row in enumerate(rows):
        log(f"Ekstraksi trial {index+1}/{len(rows)}: {row['trial_id']} ({row['label']})")
        signal,quality=prepare(row['file'],channels,row['start_s'],row['duration_s'],SETTINGS)
        x,names=features(signal,channels);xs.append(x);ys.extend([LABELS.index(row['label'])]*len(x));splits.extend([row['split']]*len(x))
        for window in range(len(x)):
            window_rows.append({**row,'start_s':row['start_s']+window*SETTINGS.window_s,'duration_s':SETTINGS.window_s})
    data={'x':np.vstack(xs),'y':np.array(ys),'split':np.array(splits),'fingerprint':np.array(fingerprint),
          'metadata':np.array(json.dumps({'channels':channels,'features':names,'labels':LABELS,'settings':SETTINGS.to_dict(),
                     'windows':window_rows,'label_source':'stimulus category; not subjective ground truth','source':SOURCE}))}
    np.savez_compressed(cache,**data)
    with (cache_dir/'trials.csv').open('w',encoding='utf-8',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    return data
