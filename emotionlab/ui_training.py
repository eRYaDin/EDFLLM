import tkinter as tk
from tkinter import ttk
from .ui_parts import paragraph,section,text_area,table


def build_training(app,parent):
    paragraph(parent,'Dua model belajar dari dua sumber berbeda. Pemaham perintah belajar dari kalimat; jaringan emosi belajar dari EEG berlabel stimulus. '
              'Checkpoint harus tersedia secara lokal. Jika belum ada, latih pemaham perintah dan jaringan EEG melalui tombol di bawah.')
    frame=section(parent,'1. Model pemaham perintah buatan sendiri')
    paragraph(frame,'154 contoh kalimat dalam corpus/commands.json; neural network ReLU 48 → 24 → 7 intent, dimulai dari bobot acak. '
              'Parser mengambil nama rekaman, emosi, jaringan, dan rentang waktu. Bukan chatbot generatif.')
    app.button(frame,'Latih ulang pemaham perintah',app.train_router).pack(anchor='w')
    app.router_status=tk.StringVar();ttk.Label(frame,textvariable=app.router_status,wraplength=1040).pack(anchor='w',pady=4)
    frame=section(parent,'2. Tujuh jaringan pengenal emosi EEG')
    paragraph(frame,'Adapter latihan saat ini: EmoEEG-MC sub-02 dan sub-03. Tujuh kategori stimulus, 84 trial. '
              'Train: sub-02 stimulus 4/5; validasi: sub-02 stimulus 8; uji: sub-03. '
              'Window dari trial yang sama tidak dibagi ke split berbeda. Ini eksperimen dua subjek, belum model emosi yang andal.')
    row=ttk.Frame(frame);row.pack(fill='x')
    ttk.Label(row,text='Epoch maksimum:').pack(side='left');app.epochs=tk.StringVar(value='160')
    app.entry(row,app.epochs,width=8).pack(side='left',padx=8)
    app.button(row,'Latih 7 jaringan · 1 vs 3 hidden layer',app.train_emotion).pack(side='left')
    app.training_status=tk.StringVar();ttk.Label(frame,textvariable=app.training_status,wraplength=1040).pack(anchor='w',pady=4)
    app.metrics_table=table(parent,{'model':'Model','depth':'Hidden layer','val':'F1 validasi','test':'F1 uji','accuracy':'Akurasi uji'},[200,140,160,160,160],height=9)
    paragraph(parent,'Sigmoid memakai tujuh kepala biner one-vs-rest. Enam eksperimen lain memakai output Softmax; '
              'eksperimen Softmax juga menguji aktivasi kompetitif di hidden layer. Jaringan dipilih dari validasi, bukan nilai uji. '
              'Banyak layer tidak otomatis lebih baik.')
