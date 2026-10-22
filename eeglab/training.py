"""Latih dari manifest berlabel, split per kelompok sebelum membentuk window."""
import csv
import json
import time
from pathlib import Path
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix
from .preprocessing import prepare, Settings
from .features import window_features
from .model_registry import create_model, MODELS
from .edf_reader import inspect_edf,common_voltage_channels


def validate_manifest_path(path):
    if path is None or not str(path).strip():
        raise ValueError('Manifest training belum dipilih. Klik "Pilih manifest CSV/TSV". '
                         'Untuk membandingkan EDF tanpa training, gunakan "Bandingkan A dan B".')
    path=Path(str(path).strip()).expanduser().resolve()
    if path.is_dir():raise ValueError('Yang dipilih adalah folder. Pilih file manifest .csv atau .tsv, bukan folder proyek.')
    if not path.is_file():raise ValueError('File manifest tidak ditemukan: '+str(path))
    if path.suffix.lower() not in ('.csv','.tsv'):
        raise ValueError('Manifest harus file .csv/.tsv, bukan EDF, JSON, atau model .npz.')
    return path


def load_manifest(path):
    path=validate_manifest_path(path)
    with path.open(encoding='utf-8-sig',newline='') as f:
        reader=csv.DictReader(f,delimiter='\t' if path.suffix.lower()=='.tsv' else ',')
        required={'file','label','group','start_s','duration_s'}
        if not required<=set(reader.fieldnames or []):
            raise ValueError('File ini belum berupa manifest training. Kolom wajib: '+', '.join(sorted(required))+
                             '. TSV penilaian perilaku (beh.tsv) tidak dapat langsung dipakai; perlu pemetaan segmen dan label.')
        rows=list(reader)
    if not rows:raise ValueError('Manifest kosong.')
    owners={};intervals={}
    for row in rows:
        for key in ('file','label','group'):row[key]=(row[key] or '').strip()
        if not all(row[k] for k in ('file','label','group')):raise ValueError('File/label/group tidak boleh kosong.')
        p=Path(row['file']);row['file']=str((path.parent/p).resolve() if not p.is_absolute() else p.resolve())
        if not Path(row['file']).is_file():raise ValueError('File tidak ditemukan: '+row['file'])
        row['start_s']=float(row['start_s']);row['duration_s']=float(row['duration_s'])
        if not np.isfinite([row['start_s'],row['duration_s']]).all() or row['start_s']<0 or row['duration_s']<4:
            raise ValueError('Segmen manifest: start >=0, duration >=4 dan harus hingga.')
        key=row['file'].casefold()
        if key in owners and owners[key]!=row['group']:raise ValueError('Satu EDF tidak boleh masuk ke dua group.')
        owners[key]=row['group']
        start=row['start_s'];end=start+row['duration_s']
        if any(start<b and end>a for a,b in intervals.get(key,[])):
            raise ValueError('Segmen dalam EDF yang sama tumpang tindih atau diduplikasi.')
        intervals.setdefault(key,[]).append((start,end))
    return rows


def resolve_training_channels(rows,requested=None):
    common=common_voltage_channels(inspect_edf(file) for file in dict.fromkeys(row['file'] for row in rows))
    if not common:raise ValueError('Tidak ada kanal tegangan bernama sama di seluruh EDF manifest.')
    if isinstance(requested,str):
        requested=None if requested.strip().lower() in ('','auto') else [c.strip() for c in requested.split(',')]
    if requested is None:return common
    if not requested or len(set(requested))!=len(requested):raise ValueError('Kanal training harus unik dan tidak kosong.')
    missing=[name for name in requested if name not in common]
    if missing:raise ValueError('Kanal training tidak tersedia dalam semua EDF manifest: '+', '.join(missing)+
                               '. Kanal yang tersedia: '+', '.join(common)+'. Gunakan auto atau pilih dari daftar ini.')
    return list(requested)


def group_split(rows,seed=42):
    y=np.array([r['label'] for r in rows]);groups=np.array([r['group'] for r in rows])
    classes=set(y)
    if len(classes)<2 or len(set(groups))<5:
        raise ValueError('Butuh minimal dua kelas dan lima group independen untuk train/validation/test.')
    for state in range(seed,seed+100):
        trainval,test=next(GroupShuffleSplit(n_splits=1,test_size=.2,random_state=state).split(y,y,groups))
        train0,val0=next(GroupShuffleSplit(n_splits=1,test_size=.25,random_state=state).split(y[trainval],y[trainval],groups[trainval]))
        train,val=trainval[train0],trainval[val0]
        if all(set(y[i])==classes for i in (train,val,test)):
            return train,val,test
    raise ValueError('Tidak bisa membagi group dengan semua kelas di setiap split. Tambah group/trial berlabel.')


def train_manifest(manifest,channels,settings,output,names=None,epochs=120,hidden=(32,16,8),log=print):
    settings.validate()
    if not 1<=epochs<=1000:raise ValueError('Epoch 1–1000.')
    rows=load_manifest(manifest);splits=group_split(rows)
    channels=resolve_training_channels(rows,channels)
    labels=sorted({r['label'] for r in rows});output=Path(output);output.mkdir(parents=True,exist_ok=True)
    datasets=[];feature_names=None
    for indexes in splits:
        xs=[];ys=[]
        for index in indexes:
            row=rows[index]
            log('Membaca '+Path(row['file']).name)
            signal,_=prepare(row['file'],channels,row['start_s'],row['duration_s'],settings)
            x,fn=window_features(signal,channels,settings)
            if feature_names is not None and fn!=feature_names:raise ValueError('Skema fitur tidak konsisten.')
            feature_names=fn;xs.append(x);ys.extend([labels.index(row['label'])]*len(x))
        datasets.append((np.vstack(xs),np.array(ys,dtype=int)))
    (xt,yt),(xv,yv),(xe,ye)=datasets
    mean=xt.mean(axis=0);std=xt.std(axis=0);std[std<1e-10]=1.
    xt=(xt-mean)/std;xv=(xv-mean)/std;xe=(xe-mean)/std
    split_info={name:sorted({rows[int(i)]['group'] for i in indexes})
                for name,indexes in zip(('train','validation','test'),splits)}
    results=[]
    for name in names or MODELS:
        if name=='sigmoid' and len(labels)!=2:
            results.append({'model':name,'status':'skipped','reason':'Sigmoid binary memerlukan tepat dua kelas.'});continue
        log('Melatih '+name)
        net=create_model(name,len(mean),len(labels),hidden)
        started=time.perf_counter()
        curves=net.fit(xt,yt,xv,yv,epochs=epochs)
        training_seconds=time.perf_counter()-started
        prediction=net.predict_proba(xe).argmax(axis=1)
        metrics={'accuracy':float(accuracy_score(ye,prediction)),
                 'balanced_accuracy':float(balanced_accuracy_score(ye,prediction)),
                 'macro_f1':float(f1_score(ye,prediction,average='macro',zero_division=0)),
                 'confusion_matrix':confusion_matrix(ye,prediction,labels=np.arange(len(labels))).tolist()}
        metadata={'schema_version':1,'model':name,'channels':list(channels),'settings':settings.to_dict(),
                  'features':feature_names,'labels':labels,'hidden':list(hidden),'seed':42,
                  'training_seconds':training_seconds,'parameter_count':sum(p.size for p in net.weights+net.biases),
                  'split_groups':split_info,'metrics_test_window':metrics,
                  'note':'Evaluasi per window; group harus subjek untuk klaim generalisasi antar-subjek.'}
        arrays={'mean':mean,'std':std,'metadata':np.array(json.dumps(metadata,ensure_ascii=False))}
        arrays.update({f'w{i}':w for i,w in enumerate(net.weights)})
        arrays.update({f'b{i}':b for i,b in enumerate(net.biases)})
        np.savez_compressed(output/(name+'.npz'),**arrays)
        results.append({'model':name,'status':'trained','metrics':metrics,'curves':curves,
                        'training_seconds':training_seconds,'parameter_count':metadata['parameter_count']})
    summary={'labels':labels,'channels':channels,'settings':settings.to_dict(),'split_groups':split_info,
             'manifest_rows':rows,'results':results,'data_kind':'labels supplied by user; no automatic emotion mapping'}
    (output/'training_report.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    return summary


def predict_file(model_path,edf,start,duration):
    with np.load(model_path,allow_pickle=False) as package:
        meta=json.loads(str(package['metadata']))
        if meta['schema_version']!=1:raise ValueError('Versi model tidak didukung.')
        settings=Settings(**meta['settings'])
        signals,_=prepare(edf,meta['channels'],start,duration,settings)
        features,names=window_features(signals,meta['channels'],settings)
        if names!=meta['features']:raise ValueError('Fitur berbeda dari pelatihan.')
        net=create_model(meta['model'],len(names),len(meta['labels']),tuple(meta['hidden']))
        net.weights=[package[f'w{i}'].copy() for i in range(len(net.weights))]
        net.biases=[package[f'b{i}'].copy() for i in range(len(net.biases))]
        standardized=(features-package['mean'])/package['std']
        if not np.isfinite(standardized).all():raise ValueError('Fitur model tidak valid.')
        proba=net.predict_proba(standardized)
    return {'file':str(edf),'labels':meta['labels'],'windows':len(proba),
            'mean_probabilities':dict(zip(meta['labels'],proba.mean(axis=0).tolist())),
            'predictions_window':[meta['labels'][i] for i in proba.argmax(axis=1)],
            'note':'Rata-rata skor model per window, bukan pengukuran kepastian atau diagnosis.'}
