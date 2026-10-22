import tkinter as tk
from tkinter import ttk
from .model_registry import MODELS
from .ui_parts import paragraph,section,table,text_area


def build_models(app,parent):
    paragraph(parent,'Bagian lanjutan. Untuk membaca dan membandingkan EDF, kamu tidak perlu training. '
              'Pilih melatih jaringan dari data berlabel atau memakai model .npz yang sudah dilatih.')
    tabs=ttk.Notebook(parent);tabs.pack(fill='both',expand=True)
    train=ttk.Frame(tabs,padding=10);tabs.add(train,text='Latih model baru')
    infer=ttk.Frame(tabs,padding=10);tabs.add(infer,text='Pakai model terlatih')
    frame=section(train,'1. Daftar data berlabel (manifest CSV/TSV)')
    paragraph(frame,'File ini harus memuat lokasi EDF, label kelas, kelompok subjek, awal dan durasi segmen. '
              'CSV hasil perbandingan dan TSV skor perilaku belum memenuhi format tersebut.')
    row=ttk.Frame(frame);row.pack(fill='x')
    app.manifest=tk.StringVar();app.entry(row,app.manifest).pack(side='left',fill='x',expand=True,padx=(0,6))
    app.button(row,'Pilih manifest',app.pick_manifest).pack(side='left')
    app.button(row,'Periksa kesiapan',app.detect_training_channels).pack(side='left',padx=5)
    app.button(row,'Buat template kosong',app.template_manifest).pack(side='left')
    app.manifest_hint=tk.StringVar(value='Belum dipilih. Training belum tersedia sampai daftar data dan label siap.')
    ttk.Label(frame,textvariable=app.manifest_hint,wraplength=850).pack(anchor='w',pady=6)
    app.manifest.trace_add('write',lambda *args:app.invalidate_manifest())
    frame=section(train,'2. Pilihan eksperimen')
    row=ttk.Frame(frame);row.pack(fill='x',pady=3)
    ttk.Label(row,text='Kanal training:').pack(side='left')
    app.training_channels=tk.StringVar(value='auto');app.entry(row,app.training_channels,width=30).pack(side='left',padx=6)
    ttk.Label(row,text='auto mengikuti EDF dalam manifest; bisa diisi nama dipisahkan koma.').pack(side='left')
    row=ttk.Frame(frame);row.pack(fill='x',pady=3)
    ttk.Label(row,text='Model:').pack(side='left')
    app.model_name=tk.StringVar(value='semua');app.combo(row,app.model_name,['semua',*MODELS],width=18).pack(side='left',padx=6)
    ttk.Label(row,text='Epoch maksimum:').pack(side='left')
    app.epochs=tk.StringVar(value='120');app.entry(row,app.epochs,width=6).pack(side='left',padx=6)
    paragraph(frame,'Semua model memakai fitur dari EDF. Yang dibandingkan adalah aktivasi dan 1 vs 3 hidden layer. '
              'Sigmoid untuk dua kelas. ReLU dan baseline Softmax memakai arsitektur yang sama. '
              'Parameter filter/sampling mengikuti langkah 2; informasi itu disimpan bersama model.')
    app.button(train,'Latih & bandingkan 1 vs 3 hidden layer',app.train,
               condition=lambda:bool(app.manifest_ready and app.manifest_ready['ready'])).pack(anchor='w',pady=6)
    app.model_result_hint=tk.StringVar(value='Hasil training akan muncul di sini. Metrik dihitung pada group pengujian yang tidak dipakai untuk training.')
    ttk.Label(train,textvariable=app.model_result_hint,wraplength=850).pack(anchor='w',pady=5)
    app.model_table=table(train,{'model':'Model','baseline':'F1 · 1 hidden','deep':'F1 · 3 hidden','note':'Status'},[150,150,150,330],height=7)
    paragraph(infer,'Diperlukan: dua EDF pada langkah 1, segmen pada langkah 2, dan model .npz hasil training aplikasi ini. '
              'Label, kanal, dan pengolahan mengikuti model tersimpan.')
    row=ttk.Frame(infer);row.pack(fill='x',pady=6)
    app.model_path=tk.StringVar();ttk.Entry(row,textvariable=app.model_path,state='readonly').pack(side='left',fill='x',expand=True,padx=(0,6))
    app.button(row,'Pilih model .npz',app.pick_model).pack(side='left')
    app.model_info=tk.StringVar(value='Belum ada model dipilih.')
    ttk.Label(infer,textvariable=app.model_info,wraplength=850).pack(anchor='w',pady=8)
    app.button(infer,'Jalankan prediksi A dan B',app.predict,
               condition=lambda:bool(app.pair and app.loaded_model)).pack(anchor='w',pady=6)
    app.prediction_text=text_area(infer)
