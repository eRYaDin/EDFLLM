"""EDF sintetis untuk memeriksa alur program; bukan data manusia/emosi."""
import csv
from datetime import date,time
from pathlib import Path
import numpy as np
import edfio


def write_signal(path,fs=160,alpha=20,beta=5,unit='uV',seed=42,seconds=16):
    rng=np.random.default_rng(seed);t=np.arange(fs*seconds)/fs
    scale=.001 if unit=='mV' else 1.
    signals=[]
    for name,gain in [('C3',1.),('C4',.8)]:
        x=gain*(alpha*np.sin(2*np.pi*10*t)+beta*np.sin(2*np.pi*20*t))+rng.normal(0,1.5,len(t))
        signals.append(edfio.EdfSignal(x*scale,fs,label=name,physical_dimension=unit,
                       physical_range=(-200*scale,200*scale)))
    record=edfio.Edf(signals,patient=edfio.Patient(code='SYNTHETIC'),
                     recording=edfio.Recording(startdate=date(2026,10,22)),starttime=time(10),
                     annotations=[edfio.EdfAnnotation(0,seconds,'SYNTHETIC_NOT_HUMAN_EEG')])
    record.write(path)


def make_demo(folder):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    write_signal(folder/'banding_A.edf',fs=160,alpha=20,beta=5)
    write_signal(folder/'banding_B.edf',fs=200,alpha=5,beta=20,unit='mV')
    rows=[]
    for group in range(10):
        for label,a,b in [('alpha_kuat',20,5),('beta_kuat',5,20)]:
            name=f'demo_g{group:02d}_{label}.edf'
            write_signal(folder/name,alpha=a,beta=b,seed=100+group*2+(label=='beta_kuat'))
            rows.append({'file':name,'label':label,'group':f'synthetic_group_{group:02d}',
                         'start_s':0,'duration_s':16})
    with (folder/'manifest.csv').open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['file','label','group','start_s','duration_s'])
        writer.writeheader();writer.writerows(rows)
    (folder/'CATATAN.txt').write_text('Seluruh EDF di folder ini adalah sinyal sinus + noise sintetis. Label alpha_kuat/beta_kuat bukan label emosi. Tidak dapat dipakai sebagai bukti akurasi pada manusia.',encoding='utf-8')
    return folder
