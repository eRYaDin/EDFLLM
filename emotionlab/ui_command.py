import tkinter as tk
from tkinter import ttk
from .ui_parts import paragraph,text_area,table


def build_commands(app,parent):
    paragraph(parent,'Tulis satu tugas EEG. Model kecil memetakan kalimat ke tindakan; pemanggil menjalankan jaringan pada sinyal. '
              'Rentang waktu mengikuti tab Data, kecuali kamu menyebut rentang seperti “detik 800 sampai 828”.')
    row=ttk.Frame(parent);row.pack(fill='x',pady=5)
    app.command=tk.StringVar(value='klasifikasikan emosi A')
    app.entry(row,app.command).pack(side='left',fill='x',expand=True,padx=(0,6))
    app.button(row,'Jalankan perintah',app.run_command).pack(side='left')
    row=ttk.Frame(parent);row.pack(fill='x',pady=5)
    for label,command in [('Klasifikasi A','klasifikasikan emosi A'),('Cari sedih','cari bagian sedih di A'),
                          ('Bandingkan emosi','bandingkan emosi A dan B'),('Bandingkan sinyal','bandingkan sinyal A dan B'),('Info EDF','tampilkan informasi EDF A')]:
        app.button(row,label,lambda c=command:app.set_command(c)).pack(side='left',padx=(0,5))
    app.route=tk.StringVar(value='Belum ada perintah dijalankan.')
    ttk.Label(parent,textvariable=app.route,wraplength=1100).pack(anchor='w',pady=6)
    tabs=ttk.Notebook(parent);tabs.pack(fill='both',expand=True)
    frame=ttk.Frame(tabs,padding=8);tabs.add(frame,text='Ringkasan hasil');app.result_text=text_area(frame)
    app.plot_frame=ttk.Frame(tabs,padding=8);tabs.add(app.plot_frame,text='Garis waktu / grafik')
    frame=ttk.Frame(tabs,padding=8);tabs.add(frame,text='Segmen & angka')
    paragraph(frame,'Belum yakin: skor rendah, selisih antar kelas kecil, atau fitur jauh dari data latihan. Kandidat tetap ditampilkan untuk pemeriksaan. Maksimal 500 baris di layar; ekspor memuat semuanya.')
    app.results_table=table(frame,{'record':'Rekaman / kanal','start':'Awal / besaran','end':'Akhir / A',
                                  'display':'Hasil / B','candidate':'Kandidat / selisih','score':'Skor'},[150,180,150,150,150,100],height=9)
    frame=ttk.Frame(tabs,padding=8);tabs.add(frame,text='Rincian proses');app.trace_text=text_area(frame)
    row=ttk.Frame(parent);row.pack(fill='x',pady=6)
    app.button(row,'Simpan hasil + CSV + grafik',app.export,condition=lambda:app.result is not None).pack(side='left')
