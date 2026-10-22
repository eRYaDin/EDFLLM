import tkinter as tk
from tkinter import ttk
from .ui_parts import paragraph,text_area,table


def build_results(app,parent):
    paragraph(parent,'Hasil di halaman ini berasal dari perhitungan sinyal. Prediksi kelas dari jaringan ditampilkan terpisah di Laboratorium model.')
    app.result_source=tk.StringVar(value='Belum ada perbandingan. Selesaikan langkah 1 dan 2.')
    ttk.Label(parent,textvariable=app.result_source,wraplength=850).pack(anchor='w',pady=(0,8))
    row=ttk.Frame(parent);row.pack(fill='x',pady=(0,8))
    ttk.Label(row,text='Lihat kanal:').pack(side='left')
    app.result_channel=tk.StringVar();app.result_selector=app.combo(row,app.result_channel,[],width=22)
    app.result_selector.pack(side='left',padx=8);app.result_selector.bind('<<ComboboxSelected>>',lambda event:app.render_result_channel())
    app.button(row,'Tanyakan hasil ini ke AI →',app.open_assistant,condition=lambda:app.result is not None).pack(side='left',padx=6)
    app.button(row,'Simpan laporan',app.save,condition=lambda:app.result is not None).pack(side='right')
    tabs=ttk.Notebook(parent);tabs.pack(fill='both',expand=True)
    frame=ttk.Frame(tabs,padding=8);tabs.add(frame,text='Penjelasan hasil')
    app.summary=text_area(frame)
    app.graph=ttk.Frame(tabs,padding=8);tabs.add(app.graph,text='Grafik sinyal & frekuensi')
    frame=ttk.Frame(tabs,padding=8);tabs.add(frame,text='Angka detail')
    paragraph(frame,'Nilai ditampilkan untuk kanal pilihan. B − A positif berarti nilai rekaman B lebih besar.')
    app.table=table(frame,{'metric':'Besaran','unit':'Satuan','a':'Rekaman A','b':'Rekaman B','delta':'Selisih B − A'},[260,80,130,130,140])


def build_assistant(app,parent):
    paragraph(parent,'AI membantu menjelaskan hasil, arti besaran, dan keterbatasan perbandingan. '
              'File EDF dibaca oleh aplikasi; LLM menerima ringkasan angka yang sudah dihitung, bukan seluruh rekaman. '
              'AI tidak melatih tujuh jaringan ketika kamu bertanya.')
    app.ai_source=tk.StringVar(value='Belum ada hasil untuk dijelaskan. Jalankan perbandingan pada langkah 2.')
    ttk.Label(parent,textvariable=app.ai_source,wraplength=850).pack(anchor='w',pady=(0,8))
    row=ttk.Frame(parent);row.pack(fill='x',pady=6)
    ttk.Label(row,text='Kanal sumber:').pack(side='left')
    app.ai_channel=tk.StringVar();app.ai_selector=app.combo(row,app.ai_channel,[],width=16)
    app.ai_selector.pack(side='left',padx=6);app.ai_selector.bind('<<ComboboxSelected>>',lambda e:app.reset_chat())
    ttk.Label(row,text='Model Ollama:').pack(side='left',padx=(12,0))
    app.llm_model=tk.StringVar(value='gemma-uncensored');app.entry(row,app.llm_model,width=24).pack(side='left',padx=6)
    app.button(row,'Periksa koneksi',app.check_llm).pack(side='left')
    app.ai_connection=tk.StringVar(value='Ollama lokal perlu aktif hanya untuk tanya AI. Analisis EDF tetap bisa berjalan tanpanya.')
    ttk.Label(parent,textvariable=app.ai_connection,wraplength=850).pack(anchor='w',pady=(0,8))
    app.llm_text=text_area(parent,height=12)
    row=ttk.Frame(parent);row.pack(fill='x',pady=6)
    for text,question in [('Jelaskan perbedaan','Jelaskan perbedaan utama A dan B pada kanal ini.'),
                           ('Arti alpha & beta','Apa arti perbedaan daya alpha dan beta dalam hasil ini?'),
                           ('Batas kesimpulan','Kesimpulan apa yang boleh dan belum boleh diambil dari hasil ini?')]:
        app.button(row,text,lambda q=question:app.ask(q),condition=lambda:app.result is not None).pack(side='left',padx=(0,6))
    row=ttk.Frame(parent);row.pack(fill='x',pady=4)
    app.question=tk.StringVar();entry=app.entry(row,app.question);entry.pack(side='left',fill='x',expand=True,padx=(0,8))
    entry.bind('<Return>',lambda event:app.guard(app.ask))
    app.button(row,'Kirim pertanyaan',app.ask,condition=lambda:app.result is not None).pack(side='left')
