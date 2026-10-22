import tkinter as tk
from tkinter import ttk
from .dataset import ROOT
from .networks import NETWORKS
from .ui_parts import paragraph,section


def build_inputs(app,parent):
    paragraph(parent,'Tujuan: prediksi kategori emosi per segmen EEG. A adalah rekaman utama; B diperlukan jika ingin membandingkan. '
              'Model perintah dibuat dan dilatih sendiri. Tidak memerlukan Ollama atau layanan chat.')
    app.paths={};app.starts={};app.durations={};app.file_info={};app.trial_boxes={}
    for key,subject in [('A','sub-02'),('B','sub-03')]:
        frame=section(parent,'Rekaman '+key+(' · wajib untuk analisis' if key=='A' else ' · opsional untuk perbandingan'))
        row=ttk.Frame(frame);row.pack(fill='x')
        path=ROOT/'data/raw'/(subject+'_task-emotion_eeg.edf')
        app.paths[key]=tk.StringVar(value=str(path) if path.is_file() else '')
        app.entry(row,app.paths[key]).pack(side='left',fill='x',expand=True,padx=(0,6))
        app.button(row,'Pilih EDF',lambda k=key:app.pick(k)).pack(side='left')
        app.button(row,'Baca file',lambda k=key:app.read_file(k)).pack(side='left',padx=5)
        app.file_info[key]=tk.StringVar(value='Klik Baca file untuk melihat kanal dan memilih percobaan dari anotasi.')
        ttk.Label(frame,textvariable=app.file_info[key],wraplength=1040).pack(anchor='w',pady=4)
        row=ttk.Frame(frame);row.pack(fill='x',pady=4)
        app.starts[key]=tk.StringVar(value='0');app.durations[key]=tk.StringVar(value='28')
        ttk.Label(row,text='Mulai (detik)').pack(side='left');app.entry(row,app.starts[key],width=12).pack(side='left',padx=6)
        ttk.Label(row,text='Durasi (detik)').pack(side='left');app.entry(row,app.durations[key],width=12).pack(side='left',padx=6)
        app.button(row,'Seluruh rekaman',lambda k=key:app.full_record(k)).pack(side='left',padx=6)
        box=app.combo(frame,tk.StringVar(),[],width=90);box.pack(fill='x',pady=3)
        app.trial_boxes[key]=box;box.bind('<<ComboboxSelected>>',lambda e,k=key:app.select_trial(k))
        app.paths[key].trace_add('write',lambda *args,k=key:app.invalidate_file(k))
        app.starts[key].trace_add('write',lambda *args:app.invalidate())
        app.durations[key].trace_add('write',lambda *args:app.invalidate())
    paragraph(parent,'Pendamping JSON/beh.tsv bernama sama dibaca saat Baca file. Label stimulus dan skor penilaian adalah informasi dataset, '
              'bukan hasil prediksi. Rekaman baru dapat dianalisis jika kanal/satuannya sesuai model yang dilatih.')
    row=ttk.Frame(parent);row.pack(fill='x',pady=8)
    ttk.Label(row,text='Jaringan EEG:').pack(side='left')
    app.network=tk.StringVar(value='auto');app.combo(row,app.network,['auto',*NETWORKS],width=18).pack(side='left',padx=8)
    app.network.trace_add('write',lambda *args:app.invalidate())
    ttk.Label(row,text='auto = pilih hasil validasi terbaik; nama jaringan dalam perintah dapat menggantikannya.').pack(side='left')
    app.button(parent,'Lanjut ke perintah →',lambda:app.tabs.select(app.command_page)).pack(anchor='w',pady=6)
