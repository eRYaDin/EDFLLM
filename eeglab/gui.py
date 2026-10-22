"""Alur terpandu EDF → segmen → hasil → penjelasan AI; training terpisah."""
import csv
import json
import queue
import threading
from datetime import datetime
from pathlib import Path
import tkinter as tk
from tkinter import ttk,filedialog,messagebox
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from .preprocessing import Settings
from .comparison import compare,export
from .training import train_manifest,predict_file,validate_manifest_path,load_manifest,resolve_training_channels
from .model_registry import MODELS
from .workflow import inspect_pair,suggested_sidecars,manifest_overview,channel_summary,METRICS,valid_edf
from .ui_parts import set_text,text_area
from .ui_data import build_data,build_analysis
from .ui_results import build_results,build_assistant
from .ui_models import build_models
from .llm import explain,list_models

ROOT=Path(__file__).resolve().parent.parent


class App:
    def __init__(self,root):
        self.root=root;self.queue=queue.Queue();self.busy=False;self.pair=None;self.result=None
        self.buttons=[];self.controls=[];self.pages={};self.nav={};self.annotation_boxes={}
        self.manifest_ready=None;self.loaded_model=None;self.chat=[];self.current_page='data';self.canvas=None
        root.title('EEG EDF Lab — Baca, Bandingkan, Pahami');root.geometry('1380x940');root.minsize(1120,780)
        root.configure(bg='#f3f5f6')
        style=ttk.Style();style.theme_use('clam')
        style.configure('.',font=('Segoe UI',10),background='#f3f5f6',foreground='#243340')
        style.configure('TButton',padding=(10,7));style.configure('TEntry',padding=5,fieldbackground='white')
        style.configure('Body.TLabel',padding=(0,2));style.configure('Title.TLabel',font=('Segoe UI',19,'bold'))
        style.configure('TLabelframe.Label',font=('Segoe UI',10,'bold'))
        style.configure('Treeview',rowheight=29,fieldbackground='white',background='white')
        style.configure('Nav.TButton',anchor='w',padding=(12,12))
        header=ttk.Frame(root,padding=(18,12));header.pack(fill='x')
        ttk.Label(header,text='EEG / EDF LAB',font=('Segoe UI',15,'bold')).pack(side='left')
        ttk.Label(header,text='Dua rekaman. Perhitungan yang jelas. Penjelasan AI bila diperlukan.').pack(side='left',padx=22)
        footer=ttk.Frame(root,padding=(16,8));footer.pack(side='bottom',fill='x')
        self.status=tk.StringVar(value='Mulai dengan memilih file EDF A dan B.')
        ttk.Label(footer,textvariable=self.status,wraplength=1000).pack(side='left',fill='x',expand=True)
        self.progress=ttk.Progressbar(footer,mode='indeterminate',length=150);self.progress.pack(side='right')
        layout=ttk.Frame(root);layout.pack(fill='both',expand=True)
        sidebar=ttk.Frame(layout,padding=(14,15),width=225);sidebar.pack(side='left',fill='y');sidebar.pack_propagate(False)
        ttk.Label(sidebar,text='ALUR UTAMA',font=('Segoe UI',9,'bold')).pack(anchor='w',pady=(0,10))
        items=[('data','1  Masukkan data'),('analysis','2  Pilih segmen'),('results','3  Lihat hasil'),('assistant','4  Tanya AI')]
        for key,label in items:
            button=ttk.Button(sidebar,text=label,style='Nav.TButton',command=lambda k=key:self.show(k))
            button.pack(fill='x',pady=3);self.nav[key]=button
        ttk.Separator(sidebar).pack(fill='x',pady=18)
        ttk.Label(sidebar,text='LANJUTAN',font=('Segoe UI',9,'bold')).pack(anchor='w')
        self.nav['models']=ttk.Button(sidebar,text='Laboratorium model',style='Nav.TButton',command=lambda:self.show('models'))
        self.nav['models'].pack(fill='x',pady=8)
        ttk.Label(sidebar,text='Tidak perlu training untuk\nmembandingkan EDF.',wraplength=190).pack(anchor='w',pady=5)
        ttk.Separator(sidebar).pack(fill='x',pady=18)
        self.button(sidebar,'Panduan file',self.help).pack(fill='x',pady=4)
        self.button(sidebar,'Data contoh (opsional)',self.demo).pack(fill='x',pady=4)
        self.content=ttk.Frame(layout,padding=(12,8,18,8));self.content.pack(side='left',fill='both',expand=True)
        self.page_title=tk.StringVar();self.breadcrumb=tk.StringVar()
        ttk.Label(self.content,textvariable=self.page_title,style='Title.TLabel').pack(anchor='w')
        ttk.Label(self.content,textvariable=self.breadcrumb).pack(anchor='w',pady=(4,14))
        holder=ttk.Frame(self.content);holder.pack(fill='both',expand=True);holder.rowconfigure(0,weight=1);holder.columnconfigure(0,weight=1)
        for key,builder in [('data',build_data),('analysis',build_analysis),('results',build_results),('assistant',build_assistant),('models',build_models)]:
            frame=ttk.Frame(holder);frame.grid(row=0,column=0,sticky='nsew');self.pages[key]=frame
            body=self.scroll_page(frame) if key in ('data','analysis') else frame
            builder(self,body)
        
        for var in self.values.values():var.trace_add('write',lambda *args:self.invalidate_analysis())
        self.channel_mode.trace_add('write',lambda *args:self.invalidate_analysis())
        self.channels.bind('<<ListboxSelect>>',lambda event:self.invalidate_analysis())
        self.show('data');root.after(100,self.poll);root.protocol('WM_DELETE_WINDOW',self.close)

    def scroll_page(self,parent):
        canvas=tk.Canvas(parent,bg='#f3f5f6',highlightthickness=0)
        bar=ttk.Scrollbar(parent,orient='vertical',command=canvas.yview);bar.pack(side='right',fill='y')
        canvas.pack(fill='both',expand=True);canvas.configure(yscrollcommand=bar.set)
        body=ttk.Frame(canvas,padding=(0,0,8,4));window=canvas.create_window(0,0,window=body,anchor='nw')
        body.bind('<Configure>',lambda e:canvas.configure(scrollregion=canvas.bbox('all')))
        canvas.bind('<Configure>',lambda e:canvas.itemconfigure(window,width=e.width))
        canvas.bind('<MouseWheel>',lambda e:canvas.yview_scroll(-int(e.delta/120),'units'))
        return body

    def button(self,parent,text,command,condition=None):
        button=ttk.Button(parent,text=text,command=lambda:self.guard(command))
        self.buttons.append((button,condition));return button

    def entry(self,parent,var,**kwargs):
        widget=ttk.Entry(parent,textvariable=var,**kwargs);self.controls.append((widget,'normal'));return widget

    def combo(self,parent,var,values,**kwargs):
        widget=ttk.Combobox(parent,textvariable=var,values=values,state='readonly',**kwargs)
        self.controls.append((widget,'readonly'));return widget

    def show(self,key):
        if self.busy:return
        if key=='analysis' and not self.pair:return
        if key in ('results','assistant') and self.result is None:return
        self.current_page=key;self.pages[key].tkraise()
        titles={'data':'1. Masukkan rekaman EEG','analysis':'2. Tentukan perbandingan','results':'3. Pahami hasil perbandingan',
                'assistant':'4. Tanyakan hasil kepada AI','models':'Laboratorium tujuh model'}
        self.page_title.set(titles[key]);self.breadcrumb.set('EDF dibaca aplikasi  →  angka dihitung  →  LLM menjelaskan hasil')
        self.refresh_controls()

    def refresh_controls(self):
        for button,condition in self.buttons:
            enabled=not self.busy and (condition is None or condition())
            button.configure(state='normal' if enabled else 'disabled')
        for control,normal in self.controls:control.configure(state='disabled' if self.busy else normal)
        for key,button in self.nav.items():
            available=key in ('data','models') or (key=='analysis' and self.pair is not None) or (key in ('results','assistant') and self.result is not None)
            button.configure(state='normal' if available and not self.busy else 'disabled')
        if hasattr(self,'channels'):
            self.channels.configure(state='disabled' if self.busy or self.channel_mode.get()=='all' else 'normal')

    def guard(self,fn):
        if self.busy:return
        try:return fn()
        except Exception as exc:self.status.set(str(exc));messagebox.showerror('Periksa langkah ini',str(exc))

    def task(self,fn,done,message='Memproses…'):
        if self.busy:return
        self.busy=True;self.status.set(message);self.progress.start(12);self.refresh_controls()
        def worker():
            try:self.queue.put(('done',done,fn()))
            except Exception as exc:self.queue.put(('error',None,str(exc)))
        threading.Thread(target=worker,daemon=True).start()

    def poll(self):
        try:
            while True:
                kind,callback,data=self.queue.get_nowait()
                if kind=='log':self.status.set(data);continue
                self.busy=False;self.progress.stop()
                if kind=='error':self.status.set('Belum berhasil: '+data);messagebox.showerror('Proses belum berhasil',data)
                else:
                    try:self.status.set('Selesai.');callback(data)
                    except Exception as exc:self.status.set(str(exc));messagebox.showerror('Tampilan',str(exc))
                self.refresh_controls()
        except queue.Empty:pass
        self.root.after(100,self.poll)

    def invalidate_analysis(self):
        if self.result is not None:
            self.clear_results();self.refresh_controls()
            if self.current_page in ('results','assistant'):self.show('analysis')

    def open_assistant(self):
        self.ai_channel.set(self.result_channel.get());self.reset_chat();self.show('assistant')

    def invalidate_inputs(self):
        self.pair=None;self.result=None;self.chat=[]
        if hasattr(self,'summary'):
            self.clear_results();self.file_overview.delete(*self.file_overview.get_children());self.refresh_controls()
            if self.current_page in ('analysis','results','assistant'):self.show('data')

    def clear_results(self):
        self.result=None;self.chat=[]
        self.result_source.set('Belum ada hasil untuk input ini. Jalankan perbandingan pada langkah 2.')
        self.ai_source.set('Belum ada hasil untuk dijelaskan.');self.result_selector.configure(values=[]);self.ai_selector.configure(values=[])
        self.result_channel.set('');self.ai_channel.set('')
        self.table.delete(*self.table.get_children());set_text(self.summary,'Hasil akan muncul setelah perbandingan selesai.')
        set_text(self.llm_text,'Selesaikan perbandingan dahulu. Pertanyaan AI akan memakai hasil yang ditampilkan.')
        for widget in self.graph.winfo_children():widget.destroy()

    def pick_edf(self,key):
        path=filedialog.askopenfilename(initialdir=ROOT/'data'/'edf',filetypes=[('Rekaman EEG','*.edf')])
        if path:
            self.paths[key].set(path)
            for kind,value in suggested_sidecars(path).items():self.sidecar_paths[key][kind].set(value)
            self.file_labels[key].set(Path(path).name+' · pendamping yang cocok ditemukan otomatis jika tersedia.')
            self.status.set('Pilih EDF kedua, lalu klik Baca file & lanjut.')

    def pick_sidecar(self,key,kind):
        path=filedialog.askopenfilename(initialdir=ROOT/'data'/'edf',filetypes=[(kind.upper(),'*.'+kind)])
        if path:self.sidecar_paths[key][kind].set(path)

    def inspect(self):
        paths={k:valid_edf(var.get()) for k,var in self.paths.items()}
        sidecars={k:{t:var.get() for t,var in values.items()} for k,values in self.sidecar_paths.items()}
        self.task(lambda:inspect_pair(paths,sidecars),self.inspected,'Membaca header, kanal, dan anotasi kedua EDF…')

    def inspected(self,pair):
        self.clear_results();self.pair=pair
        self.file_overview.delete(*self.file_overview.get_children())
        for key,info in pair['info'].items():
            rates=sorted({c['sample_rate'] for c in info['channels']})
            self.file_overview.insert('','end',values=(key,f"{info['duration_s']/60:.1f} menit",len(info['channels']),', '.join(f'{v:g}' for v in rates)+' Hz',len(info['annotations'])))
            self.file_labels[key].set(Path(info['path']).name+f" · {len(info['channels'])} kanal · {info['duration_s']:g} detik")
            annotations=info['annotations']
            self.annotation_boxes[key].configure(values=['Pilih anotasi (opsional)']+[f"{a['onset_s']:.3f} s | {a['description']}" for a in annotations])
            self.annotation_boxes[key].current(0)
            meta=pair['sidecars'][key].get('json',{}).get('data',{})
            self.values['reference_'+key.lower()].set(str(meta.get('EEGReference','')))
        self.channels.configure(state='normal');self.channels.delete(0,'end')
        for name in pair['channels']:self.channels.insert('end',name)
        self.channels.selection_set(0,'end');self.channel_mode.set('all')
        self.channel_note.set(f"{len(pair['channels'])} kanal tegangan bersama ditemukan. Seluruhnya dipakai secara default.")
        native=min(c['sample_rate'] for info in pair['info'].values() for c in info['channels'] if c['name'] in pair['channels'])
        target=min(128.,native);self.values['fs'].set(f'{target:g}');self.values['high'].set(f'{min(45.,target*.45):g}')
        duration=min(10.,*(info['duration_s'] for info in pair['info'].values()))
        self.values['start_a'].set('0');self.values['start_b'].set('0');self.values['duration'].set(f'{duration:g}')
        self.analysis_hint.set(f"File siap. {len(pair['channels'])} kanal bersama; sampling analisis awal {target:g} Hz. Pilih waktu yang sesuai kondisi yang ingin dibandingkan.")
        self.show('analysis');self.status.set('EDF terbaca. Langkah berikutnya: tentukan segmen, lalu hitung perbandingan.')

    def select_annotation(self,key):
        index=self.annotation_boxes[key].current()-1
        if self.pair and index>=0:
            onset=self.pair['info'][key]['annotations'][index]['onset_s']
            self.values['start_'+key.lower()].set(f'{max(0,onset):g}')

    def selected(self):
        if not self.pair:raise ValueError('Baca kedua EDF pada langkah 1 dahulu.')
        if self.channel_mode.get()=='all':return list(self.pair['channels'])
        channels=[self.channels.get(i) for i in self.channels.curselection()]
        if not channels:raise ValueError('Pilih setidaknya satu kanal, atau gunakan Semua kanal.')
        return channels

    def settings(self):
        settings=Settings(**{k:float(self.values[k].get()) for k in ('fs','low','high')});settings.validate();return settings

    def numbers(self):return [float(self.values[k].get()) for k in ('start_a','start_b','duration')]

    def analyze(self):
        channels=self.selected();settings=self.settings();sa,sb,duration=self.numbers()
        if not np.isfinite([sa,sb,duration]).all() or sa<0 or sb<0 or not 4<=duration<=600:
            raise ValueError('Awal segmen harus >=0; durasi 4–600 detik.')
        for key,start in [('A',sa),('B',sb)]:
            if start+duration>self.pair['info'][key]['duration_s']:raise ValueError('Segmen '+key+' melewati akhir rekaman. Kurangi awal atau durasi.')
        a,b=[self.pair['info'][k]['path'] for k in ('A','B')]
        ra,rb=[self.values[k].get() for k in ('reference_a','reference_b')]
        self.clear_results()
        self.task(lambda:compare(a,b,channels,sa,sb,duration,settings,ra,rb),self.analyzed,f'Menghitung perbandingan {len(channels)} kanal × {duration:g} detik…')

    def analyzed(self,result):
        self.result=result;report=result['report'];channels=report['channels']
        for selector in (self.result_selector,self.ai_selector):selector.configure(values=channels)
        self.result_channel.set(channels[0]);self.ai_channel.set(channels[0])
        self.result_source.set(f"{Path(report['file_A']['path']).name}  vs  {Path(report['file_B']['path']).name}\n"
                               f"{len(channels)} kanal · {report['duration_s']:g} detik · {report['settings']['fs']:g} Hz · sumber: perhitungan sinyal")
        self.ai_source.set('AI memakai hasil perbandingan terakhir dan kanal yang kamu pilih di bawah.')
        self.render_result_channel();self.reset_chat();self.show('results')
        self.status.set('Perbandingan selesai. Baca penjelasan, lihat grafik, atau lanjut ke Tanya AI.')

    def render_result_channel(self):
        if self.result is None:return
        channel=self.result_channel.get();report=self.result['report']
        set_text(self.summary,channel_summary(report,channel));self.table.delete(*self.table.get_children())
        for row in report['comparison']:
            if row['channel']!=channel:continue
            label,unit=METRICS.get(row['metric'],(row['metric'],''))
            nums=['Di luar rentang filter' if row[k] is None else f'{row[k]:.5g}' for k in ('A','B','B_minus_A')]
            self.table.insert('','end',values=[label,unit,*nums])
        for widget in self.graph.winfo_children():widget.destroy()
        self.figure=Figure(figsize=(9,6),tight_layout=True)
        i=report['channels'].index(channel);fs=report['settings']['fs'];n=min(self.result['signals_A'].shape[1],round(3*fs));time=np.arange(n)/fs
        ax=self.figure.add_subplot(211)
        for label,key in [('A','signals_A'),('B','signals_B')]:ax.plot(time,self.result[key][i,:n],label=label,lw=.8)
        ax.set(title=f'{channel} · {n/fs:g} detik pertama dari segmen analisis',xlabel='Waktu relatif segmen (s)',ylabel='Amplitudo (µV)');ax.legend();ax.grid(alpha=.2)
        ax=self.figure.add_subplot(212);f,pa,pb=self.result['spectra'][channel]
        ax.semilogy(f,np.maximum(pa,1e-20),label='A');ax.semilogy(f,np.maximum(pb,1e-20),label='B')
        ax.set(xlabel='Frekuensi (Hz)',ylabel='PSD (µV²/Hz)',xlim=(0,fs/2));ax.legend();ax.grid(alpha=.2)
        self.canvas=FigureCanvasTkAgg(self.figure,master=self.graph);self.canvas.get_tk_widget().pack(fill='both',expand=True);self.canvas.draw()

    def details(self):
        if not self.pair:return
        window=tk.Toplevel(self.root);window.title('Rincian file — sumber data');window.geometry('950x650')
        tabs=ttk.Notebook(window);tabs.pack(fill='both',expand=True)
        for key in ('A','B'):
            frame=ttk.Frame(tabs,padding=10);tabs.add(frame,text='Rekaman '+key)
            extras=self.pair['sidecars'][key]
            view={k:v for k,v in extras.items() if k!='tsv'}
            if 'tsv' in extras:view['tsv']={**extras['tsv'],'rows':extras['tsv']['rows'][:200],'display_note':'Maksimal 200 baris ditampilkan.'}
            area=text_area(frame);set_text(area,json.dumps({'dari_edf':self.pair['info'][key],'pendamping':view,
                        'catatan':self.pair['warnings']},ensure_ascii=False,indent=2))

    def save(self):
        if self.result is None:return
        folder=filedialog.askdirectory(title='Pilih folder untuk hasil analisis')
        if folder:
            result=self.result;path=Path(folder)/('edf-'+datetime.now().strftime('%Y%m%d-%H%M%S-%f'))
            self.task(lambda:export(result,path),lambda p:messagebox.showinfo('Laporan tersimpan','JSON: seluruh hasil dan metadata\nCSV: tabel perbandingan\nPNG: grafik kanal pertama\n\n'+str(p)))

    def reset_chat(self):
        self.chat=[]
        if self.result is not None:set_text(self.llm_text,'Sumber: hasil hitung kanal '+self.ai_channel.get()+'.\nPilih pertanyaan contoh atau tulis pertanyaanmu.\nPergantian kanal/perbandingan memulai percakapan baru.')

    def ask(self,question=None):
        if self.result is None:return
        question=(question or self.question.get()).strip()
        if not question:raise ValueError('Tulis pertanyaan tentang hasil atau pilih salah satu contoh.')
        if len(question)>2000:raise ValueError('Pertanyaan maksimal 2000 karakter.')
        channel=self.ai_channel.get();report=self.result['report'];model=self.llm_model.get().strip();history=list(self.chat)
        if not model:raise ValueError('Isi nama model Ollama yang tersedia.')
        def done(text):
            self.chat.extend([{'role':'user','content':question},{'role':'assistant','content':text}]);self.chat=self.chat[-6:]
            self.question.set('');set_text(self.llm_text,'Sumber: hitungan kanal '+channel+'\n\n'+'\n\n'.join(('Fif: ' if m['role']=='user' else 'AI: ')+m['content'] for m in self.chat))
            self.status.set('Penjelasan AI selesai. Angka acuan tetap ada di halaman Hasil.')
        self.task(lambda:explain(report,model,question=question,channel=channel,history=history),done,'Ollama sedang menjelaskan hasil; file EDF tidak dikirim ke layanan cloud…')

    def check_llm(self):
        self.task(list_models,lambda names:self.ai_connection.set('Ollama terhubung. Model tersedia: '+', '.join(names)),
                  'Memeriksa Ollama lokal…')

    def invalidate_manifest(self):
        self.manifest_ready=None
        if hasattr(self,'manifest_hint'):self.manifest_hint.set('Klik Periksa kesiapan sebelum training. File harus berisi daftar segmen berlabel.')
        if hasattr(self,'model_path'):self.refresh_controls()

    def pick_manifest(self):
        path=filedialog.askopenfilename(initialdir=ROOT/'data',filetypes=[('Daftar training','*.csv *.tsv')])
        if path:self.manifest.set(path);self.training_channels.set('auto');self.detect_training_channels()

    def detect_training_channels(self):
        path=validate_manifest_path(self.manifest.get())
        def done(info):
            self.manifest_ready=info
            text=f"{info['files']} EDF · {info['groups']} group · {info['rows']} segmen · {len(info['channels'])} kanal bersama\nLabel: "+', '.join(info['labels'])
            text+='\nSiap dilatih.' if info['ready'] else '\nBelum siap: '+info['reason']
            self.manifest_hint.set(text)
        self.task(lambda:manifest_overview(path),done,'Memeriksa daftar file, kanal, label, dan pemisahan group…')

    def template_manifest(self):
        path=filedialog.asksaveasfilename(initialdir=ROOT/'data',initialfile='manifest_training.csv',defaultextension='.csv')
        if path:
            with open(path,'w',encoding='utf-8',newline='') as file:csv.writer(file).writerow(['file','label','group','start_s','duration_s'])
            messagebox.showinfo('Template dibuat','Isi satu baris per segmen dengan label dan waktu yang sudah diverifikasi. Template ini belum bisa dilatih sebelum diisi.\n\n'+path)

    def train(self):
        if not self.manifest_ready or not self.manifest_ready['ready']:raise ValueError('Periksa kesiapan manifest dahulu.')
        manifest=validate_manifest_path(self.manifest.get());settings=self.settings();epochs=int(self.epochs.get());requested=self.training_channels.get()
        if not 1<=epochs<=1000:raise ValueError('Epoch maksimum 1–1000.')
        chosen=self.model_name.get();names=list(MODELS) if chosen=='semua' else [chosen]
        output=ROOT/'output'/'models'/datetime.now().strftime('%Y%m%d-%H%M%S-%f')
        def work():
            channels=resolve_training_channels(load_manifest(manifest),requested);reports={}
            for depth,hidden in [('baseline',(32,)),('deep',(32,16,8))]:
                reports[depth]=train_manifest(manifest,channels,settings,output/depth,names,epochs,hidden,
                    log=lambda text:self.queue.put(('log',None,text)))
            return reports
        def done(reports):
            self.model_table.delete(*self.model_table.get_children())
            for name in names:
                values=[];note='Terlatih'
                for depth in ('baseline','deep'):
                    row=next(r for r in reports[depth]['results'] if r['model']==name)
                    values.append(f"{row['metrics']['macro_f1']:.3f}" if row['status']=='trained' else 'Dilewati')
                    if row['status']!='trained':note=row['reason']
                self.model_table.insert('','end',values=(name,*values,note))
            self.model_result_hint.set('Selesai. Model .npz dan laporan training disimpan di '+str(output))
        self.task(work,done,'Melatih model dari manifest yang sudah diperiksa…')

    def pick_model(self):
        path=filedialog.askopenfilename(initialdir=ROOT/'output'/'models',filetypes=[('Model hasil training','*.npz')])
        if not path:return
        with np.load(path,allow_pickle=False) as package:meta=json.loads(str(package['metadata']))
        if meta.get('schema_version')!=1:raise ValueError('Model tidak didukung.')
        self.loaded_model=meta;self.model_path.set(path)
        self.model_info.set(f"{meta['model']} · label: {', '.join(meta['labels'])}\n{len(meta['channels'])} kanal: "+', '.join(meta['channels'])+
                            '\nModel hanya memprediksi label tersebut. Data demo tidak membuktikan kemampuan mengenali emosi manusia.')
        self.refresh_controls()

    def predict(self):
        if not self.pair or not self.loaded_model:return
        path=self.model_path.get();sa,sb,duration=self.numbers();a,b=[self.pair['info'][key]['path'] for key in ('A','B')]
        def done(result):
            lines=['HASIL MODEL TERLATIH — terpisah dari perhitungan perbandingan']
            for key in ('A','B'):
                values=result[key];lines+=['',f"Rekaman {key} · {values['windows']} window"]
                lines += [f"{label}: skor rata-rata {score:.4f}" for label,score in values['mean_probabilities'].items()]
            lines+=['','Skor model bukan kepastian emosi atau diagnosis. Kanal dan pengolahan mengikuti model .npz.']
            set_text(self.prediction_text,'\n'.join(lines))
        self.task(lambda:{'A':predict_file(path,a,sa,duration),'B':predict_file(path,b,sb,duration)},done,'Menjalankan model terlatih pada EDF A dan B…')

    def help(self):
        messagebox.showinfo('File apa yang diperlukan?',
            'ANALISIS BIASA\nWajib: EDF A dan EDF B (boleh file yang sama).\nOpsional: JSON metadata dan TSV pendamping.\nTidak perlu manifest atau model terlatih.\n\n'
            'TANYA AI\nSelesaikan perbandingan, lalu jalankan Ollama lokal. AI menjelaskan hasil hitungan.\n\n'
            'TRAINING LANJUTAN\nPerlu manifest CSV/TSV berisi file,label,group,start_s,duration_s.\nperbandingan.csv dan beh.tsv bukan manifest.\n\n'
            'PREDIKSI LANJUTAN\nPerlu model .npz yang sudah dilatih dan EDF dengan kanal yang sesuai.')

    def demo(self):
        if not messagebox.askyesno('Data contoh sintetis','Beralih ke data sintetis untuk latihan? Ini akan mengganti EDF A/B pilihan saat ini. Untuk data asli, pilih Tidak.'):return
        folder=ROOT/'demo'
        if not (folder/'banding_A.edf').exists():raise ValueError('Demo belum dibuat. Jalankan BUAT_DEMO.bat dahulu.')
        for key in ('A','B'):
            self.paths[key].set(str(folder/f'banding_{key}.edf'))
            for var in self.sidecar_paths[key].values():var.set('')
        self.manifest.set(str(folder/'manifest.csv'));self.training_channels.set('auto');self.inspect()

    def close(self):
        if self.busy:self.status.set('Tunggu proses selesai sebelum menutup.');return
        self.root.destroy()


def main():
    root=tk.Tk();App(root);root.mainloop()
