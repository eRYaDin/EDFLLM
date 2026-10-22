import tkinter as tk
from tkinter import ttk
from .ui_parts import paragraph,section,table


def build_data(app,parent):
    paragraph(parent,'Mulai dengan dua file .edf. Aplikasi membaca isi rekaman dan menghitung perbedaannya. '
              'File yang sama boleh dipakai sebagai A dan B untuk membandingkan bagian waktu yang berbeda.')
    app.paths={};app.sidecar_paths={};app.file_labels={}
    for key in ('A','B'):
        frame=section(parent,'Rekaman '+key+' · wajib EDF')
        row=ttk.Frame(frame);row.pack(fill='x')
        app.paths[key]=tk.StringVar()
        app.entry(row,app.paths[key]).pack(side='left',fill='x',expand=True,padx=(0,8))
        app.button(row,'Pilih EDF '+key,lambda k=key:app.pick_edf(k)).pack(side='left')
        app.file_labels[key]=tk.StringVar(value='Belum dipilih. JSON dan TSV tidak menggantikan file sinyal EDF.')
        ttk.Label(frame,textvariable=app.file_labels[key],wraplength=850).pack(anchor='w',pady=(5,0))
        app.sidecar_paths[key]={kind:tk.StringVar() for kind in ('json','tsv')}
        app.paths[key].trace_add('write',lambda *args:app.invalidate_inputs())
    app.extras_open=tk.BooleanVar(value=False)
    extras=ttk.Frame(parent)
    def toggle():
        if app.extras_open.get():extras.pack(fill='x',after=app.extras_toggle,pady=5)
        else:extras.pack_forget()
    app.extras_toggle=ttk.Checkbutton(parent,text='Lihat file pendamping opsional (JSON / TSV)',variable=app.extras_open,command=toggle)
    app.extras_toggle.pack(anchor='w',pady=(0,8))
    paragraph(extras,'JSON membantu menjelaskan pengaturan rekaman. TSV menampilkan tabel kejadian atau penilaian sesuai isi file. '
              'Pendamping yang namanya cocok di folder yang sama akan ditemukan otomatis. Tidak dipakai sebagai manifest training.')
    for key in ('A','B'):
        for kind in ('json','tsv'):
            row=ttk.Frame(extras);row.pack(fill='x',pady=2)
            ttk.Label(row,text=f'{key} · {kind.upper()}',width=10).pack(side='left')
            app.entry(row,app.sidecar_paths[key][kind]).pack(side='left',fill='x',expand=True,padx=5)
            app.button(row,'Pilih',lambda k=key,t=kind:app.pick_sidecar(k,t)).pack(side='left')
            app.sidecar_paths[key][kind].trace_add('write',lambda *args:app.invalidate_inputs())
    app.file_overview=table(parent,{'record':'Rekaman','duration':'Durasi','channels':'Kanal sinyal',
                                 'sampling':'Sampling asli','events':'Anotasi EDF'},height=3)
    row=ttk.Frame(parent);row.pack(fill='x',pady=10)
    app.read_button=app.button(row,'Baca file & lanjut ke pemilihan segmen',app.inspect)
    app.read_button.pack(side='left')
    app.button(row,'Lihat rincian file & tabel pendamping',app.details,condition=lambda:bool(app.pair)).pack(side='left',padx=8)


def build_analysis(app,parent):
    paragraph(parent,'Tentukan bagian rekaman yang ingin dibandingkan. Pilihan kanal diambil dari file EDF, bukan dari daftar tetap.')
    app.analysis_hint=tk.StringVar(value='Masukkan EDF A dan B pada langkah 1 terlebih dahulu.')
    ttk.Label(parent,textvariable=app.analysis_hint,wraplength=850).pack(anchor='w',pady=(0,10))
    frame=section(parent,'Bagian waktu yang dibandingkan')
    app.values={}
    for key,label,value in [('start_a','Mulai A (detik)','0'),('start_b','Mulai B (detik)','0'),('duration','Lama segmen (detik)','10')]:
        row=ttk.Frame(frame);row.pack(fill='x',pady=3)
        ttk.Label(row,text=label,width=25).pack(side='left')
        app.values[key]=tk.StringVar(value=value);app.entry(row,app.values[key],width=12).pack(side='left')
        if key!='duration':
            which='A' if key=='start_a' else 'B'
            combo=app.combo(row,tk.StringVar(),[],width=45);combo.pack(side='left',padx=8)
            app.annotation_boxes[which]=combo
            combo.bind('<<ComboboxSelected>>',lambda event,k=which:app.select_annotation(k))
    paragraph(frame,'Anotasi dapat membantu memilih waktu. Trigger yang sama/berbeda belum tentu berarti label emosi yang sama.')
    frame=section(parent,'Kanal yang dianalisis')
    app.channel_note=tk.StringVar(value='Kanal akan tersedia setelah EDF dibaca.')
    ttk.Label(frame,textvariable=app.channel_note,wraplength=850).pack(anchor='w')
    app.channel_mode=tk.StringVar(value='all')
    row=ttk.Frame(frame);row.pack(fill='x',pady=5)
    for text,value in [('Semua kanal tegangan bersama','all'),('Pilih sebagian kanal','custom')]:
        button=ttk.Radiobutton(row,text=text,value=value,variable=app.channel_mode,command=app.refresh_controls)
        button.pack(side='left',padx=(0,12));app.controls.append((button,'normal'))
    app.channels=tk.Listbox(frame,selectmode='extended',exportselection=False,height=4)
    app.channels.pack(fill='x');app.controls.append((app.channels,'normal'))
    paragraph(frame,'Untuk pilihan sebagian, gunakan Ctrl/Shift. Nama dan satuan harus cocok; kanal tanpa satuan tegangan tidak diikutkan.')
    advanced=section(parent,'Pengaturan pengolahan · default bisa dipakai untuk mulai')
    for key,label,value in [('fs','Sampling analisis (Hz)','128'),('low','Batas bawah filter (Hz)','1'),('high','Batas atas filter (Hz)','45'),
                            ('reference_a','Referensi elektroda A',''),('reference_b','Referensi elektroda B','')]:
        row=ttk.Frame(advanced);row.pack(fill='x',pady=2)
        ttk.Label(row,text=label,width=27).pack(side='left')
        app.values[key]=tk.StringVar(value=value);app.entry(row,app.values[key],width=22).pack(side='left')
    paragraph(advanced,'Sampling dapat diturunkan agar kedua file sebanding. Referensi dapat terisi dari JSON; mengisi kolom ini tidak mengubah montase sinyal.')
    row=ttk.Frame(parent);row.pack(fill='x',pady=5)
    app.button(row,'Hitung perbandingan → lihat hasil',app.analyze,condition=lambda:bool(app.pair)).pack(side='left')
