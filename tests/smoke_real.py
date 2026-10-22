"""Run from project root: python -m tests.smoke_real (needs the two local EDFs)."""
import json
import tkinter as tk
from emotionlab.dataset import ROOT,build_manifest,prepare_dataset
from emotionlab.dispatcher import dispatch
from emotionlab.gui import App

rows,channels=build_manifest();assert len(rows)==84 and len(channels)==64
data=prepare_dataset(log=lambda text:None);meta=json.loads(str(data['metadata']))
assert all(r['subject']=='sub-03' for r in meta['windows'] if r['split']=='test')
assert all(r['subject']=='sub-02' for r in meta['windows'] if r['split']!='test')
records={}
for key,subject in [('A','sub-02'),('B','sub-03')]:
    trial=next(r for r in rows if r['subject']==subject)
    records[key]={'file':trial['file'],'start':trial['start_s'],'duration':28}
root=tk.Tk();root.withdraw();app=App(root)
for command,kind in [('klasifikasikan emosi A','classify'),('cari bagian sedih','find'),
                      ('bandingkan emosi A dan B','compare_emotion'),('bandingkan sinyal A dan B','signal'),
                      ('tampilkan informasi EDF A','metadata')]:
    result=dispatch(command,records,log=lambda text:None);assert result['kind']==kind
    app.render(result);root.update();print('PASS',command)
app.starts['A'].set('4');assert app.result is None
root.destroy();print('PASS 84 trials, dynamic 64 channels, independent subject test, GUI and stale result reset')
