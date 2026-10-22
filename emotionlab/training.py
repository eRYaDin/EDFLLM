"""Train tujuh jaringan dari EEG berlabel stimulus; pilih model hanya dari validasi."""
import json
import time
from pathlib import Path
import numpy as np
from sklearn.metrics import accuracy_score,f1_score,confusion_matrix
from .networks import NETWORKS,create
from .dataset import ROOT,prepare_dataset,LABELS


def metrics(y,scores):
    prediction=scores.argmax(axis=1)
    return {'accuracy':float(accuracy_score(y,prediction)),
            'macro_f1':float(f1_score(y,prediction,labels=range(len(LABELS)),average='macro',zero_division=0)),
            'confusion_matrix':confusion_matrix(y,prediction,labels=range(len(LABELS))).tolist()}


def train_all(raw_dir=ROOT/'data/raw',output=ROOT/'models/emotion',epochs=160,log=print):
    if not 1<=epochs<=1000:raise ValueError('Epoch 1–1000.')
    data=prepare_dataset(raw_dir,log=log);meta=json.loads(str(data['metadata']))
    x,y,split=data['x'],data['y'],data['split'];ix={k:np.where(split==k)[0] for k in ('train','validation','test')}
    if any(set(y[index])!=set(range(len(LABELS))) for index in ix.values()):raise ValueError('Setiap split perlu semua kelas.')
    mean=x[ix['train']].mean(axis=0);std=x[ix['train']].std(axis=0);std[std<1e-8]=1.
    z=(x-mean)/std;output=Path(output);output.mkdir(parents=True,exist_ok=True);results=[]
    for name in NETWORKS:
        for depth,hidden in [('baseline',(32,)),('deep',(32,16,8))]:
            log('Melatih '+name+' / '+depth);net=create(name,x.shape[1],len(LABELS),hidden);start=time.perf_counter()
            curves=net.fit(z[ix['train']],y[ix['train']],z[ix['validation']],y[ix['validation']],epochs=epochs)
            val=metrics(y[ix['validation']],net.predict_scores(z[ix['validation']]))
            test=metrics(y[ix['test']],net.predict_scores(z[ix['test']]))
            model_id=name+'_'+depth
            model_meta={'schema':2,'id':model_id,'network':name,'hidden':list(hidden),
                        'channels':meta['channels'],'features':meta['features'],'settings':meta['settings'],
                        'labels':LABELS,'label_source':meta['label_source'],'dataset_fingerprint':str(data['fingerprint']),
                        'train_subjects':['sub-02'],'validation_subjects':['sub-02'],'test_subjects':['sub-03'],
                        'validation':val,'test':test,'independent_sigmoid':net.independent_sigmoid,
                        'training_seconds':time.perf_counter()-start,
                        'parameters':sum(a.size for a in net.weights+net.biases)}
            np.savez_compressed(output/(model_id+'.npz'),mean=mean,std=std,metadata=np.array(json.dumps(model_meta)),**net.arrays())
            results.append({**model_meta,'curves':curves})
    best=max(results,key=lambda r:r['validation']['macro_f1'])
    counts={key:int(len(value)) for key,value in ix.items()}
    report={'schema':2,'selected':best['id'],'selection_rule':'highest validation macro_f1; test never used to select',
            'labels':LABELS,'split_windows':counts,'split_policy':'train sub-02 stimulus 4/5; validation sub-02 stimulus 8; test sub-03',
            'limitations':['Hanya dua subjek; satu subjek uji, belum validasi populasi.',
                           'Label kategori stimulus, bukan kepastian perasaan pada setiap window.',
                           'Tidak ada ICA atau model artefak tervalidasi; fitur awal bandpower/RMS.',
                           'Skor belum dikalibrasi. Ketidakpastian ditandai dengan ambang heuristik.'],
            'results':results}
    (output/'training_report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
    return report


def load_model(path):
    with np.load(path,allow_pickle=False) as package:
        meta=json.loads(str(package['metadata']))
        if meta.get('schema')!=2:raise ValueError('Gunakan model emosi versi baru; model pembanding lama tidak cocok.')
        net=create(meta['network'],len(meta['features']),len(meta['labels']),tuple(meta['hidden']));net.restore(package)
        mean=package['mean'].copy();std=package['std'].copy()
    if mean.shape!=(net.inputs,) or std.shape!=mean.shape or not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std<=0):
        raise ValueError('Normalisasi model rusak.')
    return net,meta,mean,std
