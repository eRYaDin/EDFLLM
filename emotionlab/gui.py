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
from .dataset import ROOT,LABELS,trials_from_info
from .edf import inspect_edf
from .commands import train_commands
from .training import train_all
from .dispatcher import dispatch
from .presentation import explain,timeline_rows,export_result
from .ui_parts import set_text
from .ui_input import build_inputs
from .ui_command import build_commands
from .ui_training import build_training


class App:
    def __init__(self,root):
        self.root=root;self.queue=queue.Queue();self.busy=False;self.result=None;self.infos={};self.trials={}
        self.buttons=[];self.controls=[];self.figure=None;self.canvas=None
        root.title('EDFLLM · Klasifikasi Emosi EEG');root.geometry('1360x930');root.minsize(1120,760)
        style=ttk.Style();style.theme_use('clam');style.configure('.',font=('Segoe UI',10))
        style.configure('TButton',padding=(10,7));style.configure('TEntry',padding=5)
        style.configure('Treeview',rowheight=28);style.configure('TLabelframe.Label',font=('Segoe UI',11,'bold'))
        header=ttk.Frame(root,padding=(18,10));header.pack(fill='x')
        ttk.Label(header,text='EDFLLM  /  EMOSI EEG',font=('Segoe UI',18,'bold')).pack(anchor='w')
        ttk.Label(header,text='Perintah → model pemaham maksud → jaringan EEG → prediksi emosi sepanjang waktu').pack(anchor='w',pady=3)
        footer=ttk.Frame(root,padding=10);footer.pack(side='bottom',fill='x')
        self.status=tk.StringVar(value='Mulai: tab Data → Baca file A/B → pilih trial atau waktu → Perintah & hasil.')
        ttk.Label(footer,textvariable=self.status,wraplength=1100).pack(side='left',fill='x',expand=True)
        self.progress=ttk.Progressbar(footer,mode='indeterminate',length=120);self.progress.pack(side='right')
        self.tabs=ttk.Notebook(root);self.tabs.pack(fill='both',expand=True,padx=14,pady=6)
        self.data_page=ttk.Frame(self.tabs,padding=12);self.tabs.add(self.data_page,text='1 · Data & rentang waktu')
        self.command_page=ttk.Frame(self.tabs,padding=12);self.tabs.add(self.command_page,text='2 · Perintah & hasil')
        self.training_page=ttk.Frame(self.tabs,padding=12);self.tabs.add(self.training_page,text='3 · Model & pelatihan')
        build_inputs(self,self.scroll(self.data_page));build_commands(self,self.command_page);build_training(self,self.scroll(self.training_page))
        self.load_metrics();self.refresh();root.after(100,self.poll);root.protocol('WM_DELETE_WINDOW',self.close)

    def scroll(self,parent):
        canvas=tk.Canvas(parent,highlightthickness=0);bar=ttk.Scrollbar(parent,command=canvas.yview)
        bar.pack(side='right',fill='y');canvas.pack(fill='both',expand=True);canvas.configure(yscrollcommand=bar.set)
        body=ttk.Frame(canvas);window=canvas.create_window(0,0,window=body,anchor='nw')
        body.bind('<Configure>',lambda e:canvas.configure(scrollregion=canvas.bbox('all')))
        canvas.bind('<Configure>',lambda e:canvas.itemconfigure(window,width=e.width));return body

    def button(self,parent,text,command,condition=None):
        widget=ttk.Button(parent,text=text,command=lambda:self.guard(command));self.buttons.append((widget,condition));return widget

    def entry(self,parent,var,**kwargs):
        widget=ttk.Entry(parent,textvariable=var,**kwargs);self.controls.append((widget,'normal'));return widget

    def combo(self,parent,var,values,**kwargs):
        widget=ttk.Combobox(parent,textvariable=var,values=values,state='readonly',**kwargs);self.controls.append((widget,'readonly'));return widget

    def refresh(self):
        for widget,condition in self.buttons:widget.configure(state='normal' if not self.busy and (condition is None or condition()) else 'disabled')
        for widget,state in self.controls:widget.configure(state='disabled' if self.busy else state)

    def guard(self,fn):
        if self.busy:return
        try:return fn()
        except Exception as exc:self.status.set(str(exc));messagebox.showerror('Periksa input',str(exc))

    def task(self,fn,done,text):
        if self.busy:return
        self.busy=True;self.status.set(text);self.progress.start(12);self.refresh()
        def work():
            try:self.queue.put(('done',done,fn()))
            except Exception as exc:self.queue.put(('error',None,str(exc)))
        threading.Thread(target=work,daemon=True).start()

    def log(self,text):self.queue.put(('log',None,text))

    def poll(self):
        try:
            while True:
                kind,callback,value=self.queue.get_nowait()
                if kind=='log':self.status.set(value);continue
                self.busy=False;self.progress.stop()
                if kind=='error':self.status.set(value);messagebox.showerror('Proses belum berhasil',value)
                else:
                    try:callback(value)
                    except Exception as exc:self.status.set(str(exc));messagebox.showerror('Tampilan',str(exc))
                self.refresh()
        except queue.Empty:pass
        self.root.after(100,self.poll)

    def invalidate(self):
        self.result=None
        if hasattr(self,'result_text'):
            set_text(self.result_text,'Input berubah. Jalankan perintah untuk hasil baru.');set_text(self.trace_text,'')
            self.results_table.delete(*self.results_table.get_children());self.route.set('Belum ada hasil untuk input ini.')
            for child in self.plot_frame.winfo_children():child.destroy()
            self.figure=None;self.refresh()

    def invalidate_file(self,key):
        self.infos.pop(key,None);self.trials.pop(key,None);self.file_info[key].set('Klik Baca file setelah memilih EDF.')
        self.trial_boxes[key].configure(values=[]);self.trial_boxes[key].set('');self.invalidate()

    def pick(self,key):
        path=filedialog.askopenfilename(initialdir=ROOT/'data/raw',filetypes=[('Sinyal EEG','*.edf')])
        if path:self.paths[key].set(path);self.starts[key].set('0');self.read_file(key)

    def read_file(self,key):
        path=Path(self.paths[key].get())
        if not path.is_file() or path.suffix.lower()!='.edf':raise ValueError('Pilih file EDF '+key+'.')
        def work():
            info=inspect_edf(path);subject=path.name.split('_')[0];trials=[];note=''
            if subject in ('sub-02','sub-03'):
                try:trials=trials_from_info(info,subject)
                except ValueError as exc:note=str(exc)
            extras=[]
            if path.with_suffix('.json').is_file():
                try:
                    meta=json.loads(path.with_suffix('.json').read_text(encoding='utf-8-sig'))
                    extras.append('JSON: referensi '+str(meta.get('EEGReference','tidak diketahui')))
                except Exception:extras.append('JSON tidak dapat dibaca')
            beh=path.with_name(path.stem.replace('_eeg','_beh')+'.tsv')
            if beh.is_file():
                with beh.open(encoding='utf-8-sig',newline='') as f:extras.append(f'TSV: {len(list(csv.DictReader(f,delimiter=chr(9))))} penilaian')
            return info,trials,note,extras
        def done(value):
            info,trials,note,extras=value;self.infos[key]=info;self.trials[key]=trials
            self.file_info[key].set(f"{len(info['channels'])} kanal | {info['duration_s']/60:.1f} menit | {len(info['annotations'])} anotasi | "+'; '.join(extras)+(' | '+note if note else ''))
            box=self.trial_boxes[key]
            box.configure(values=[f"{r['context']} / {r['stimulus']} / label stimulus {r['label']} / {r['start_s']:.2f}–{r['start_s']+r['duration_s']:.2f}s" for r in trials])
            if trials:box.current(0);self.select_trial(key)
            else:box.set('Tidak ada profil label otomatis; isi awal/durasi sendiri.')
            self.status.set('EDF '+key+' terbaca. Label pada pemilih trial adalah label stimulus, bukan prediksi.')
        self.task(work,done,'Membaca EDF '+key+' dan pendamping…')

    def select_trial(self,key):
        i=self.trial_boxes[key].current()
        if i>=0 and i<len(self.trials.get(key,[])):
            trial=self.trials[key][i];self.starts[key].set(f"{trial['start_s']:.4f}");self.durations[key].set(f"{trial['duration_s']:.4f}")

    def full_record(self,key):
        if key not in self.infos:raise ValueError('Klik Baca file '+key+' dahulu.')
        self.starts[key].set('0');self.durations[key].set(str(self.infos[key]['duration_s']))
        self.status.set('Seluruh rekaman dipilih, termasuk jeda dan penilaian yang tidak termasuk data latihan. Proses dapat memakan beberapa menit.')

    def set_command(self,text):self.command.set(text)

    def run_command(self):
        text=self.command.get().strip();network=self.network.get()
        records={k:{'file':self.paths[k].get(),'start':self.starts[k].get(),'duration':self.durations[k].get()} for k in ('A','B')}
        self.invalidate();self.task(lambda:dispatch(text,records,network,log=self.log),self.render,'Memahami perintah, memeriksa input, lalu menjalankan analisis…')

    def render(self,result):
        self.result=result;plan=result['plan']
        model_names=', '.join(dict.fromkeys(t['model'] for t in result.get('timelines',{}).values()))
        self.route.set(f"Maksud: {plan['intent']} | skor pengenal: {plan['score']:.3f}"+(' | jaringan dijalankan: '+model_names if model_names else ''))
        set_text(self.result_text,explain(result));set_text(self.trace_text,json.dumps(plan,ensure_ascii=False,indent=2))
        self.results_table.delete(*self.results_table.get_children())
        for row in timeline_rows(result)[:500]:
            self.results_table.insert('','end',values=(row['record'],f"{row['start_s']:.2f}s",f"{row['end_s']:.2f}s",row['display'],row['prediction'],f"{row['score']:.3f}"))
        if result['kind']=='signal':
            for row in result['result']['report']['comparison'][:500]:
                self.results_table.insert('','end',values=(row['channel'],row['metric'],*[f'{row[k]:.4g}' if row[k] is not None else 'n/a' for k in ('A','B','B_minus_A')],''))
        self.draw(result);self.status.set('Selesai. Baca ringkasan, grafik dan metrik; kandidat emosi bukan kepastian.');self.refresh()

    def draw(self,result):
        for child in self.plot_frame.winfo_children():child.destroy()
        self.figure=None
        if not result.get('timelines') and result['kind']!='signal':return
        fig=Figure(figsize=(10,5),tight_layout=True)
        if result['kind']=='signal':
            value=result['result'];report=value['report'];ch=report['channels'][0]
            ax=fig.add_subplot(211);n=min(len(value['signals_A'][0]),int(5*report['settings']['fs']));t=np.arange(n)/report['settings']['fs']
            ax.plot(t,value['signals_A'][0,:n],label='A');ax.plot(t,value['signals_B'][0,:n],label='B')
            ax.set(title=ch+' · 5 detik pertama',xlabel='Waktu relatif (s)',ylabel='µV');ax.legend()
            ax=fig.add_subplot(212);freq,pa,pb=value['spectra'][ch];ax.semilogy(freq,np.maximum(pa,1e-20),label='A');ax.semilogy(freq,np.maximum(pb,1e-20),label='B')
            ax.set(xlabel='Frekuensi (Hz)',ylabel='PSD (µV²/Hz)');ax.legend()
        else:
            for index,(key,timeline) in enumerate(result['timelines'].items(),1):
                ax=fig.add_subplot(len(result['timelines']),1,index);rows=timeline['rows'];labels=[*LABELS,'belum yakin']
                ax.scatter([(r['start_s']+r['end_s'])/2 for r in rows],[labels.index(r['display']) for r in rows],s=18,c=['#9ca3af' if r['display']=='belum yakin' else '#296b83' for r in rows])
                ax.set(yticks=range(len(labels)),yticklabels=labels,xlabel='Waktu rekaman (detik)',title=f'Rekaman {key} · prediksi {timeline["model"]}');ax.grid(alpha=.2)
        self.figure=fig;self.canvas=FigureCanvasTkAgg(fig,master=self.plot_frame);self.canvas.get_tk_widget().pack(fill='both',expand=True);self.canvas.draw()

    def export(self):
        folder=filedialog.askdirectory(title='Folder hasil')
        if not folder or self.result is None:return
        target=Path(folder)/datetime.now().strftime('emosi-%Y%m%d-%H%M%S-%f')
        export_result(self.result,target)
        if self.figure:self.figure.savefig(target/'grafik.png',dpi=150)
        self.status.set('Hasil disimpan: '+str(target))

    def train_router(self):
        self.task(train_commands,lambda report:self.training_done(),'Melatih model kecil dari corpus kalimat…')

    def train_emotion(self):
        epochs=int(self.epochs.get())
        if not 1<=epochs<=1000:raise ValueError('Epoch 1–1000.')
        self.invalidate();self.task(lambda:train_all(epochs=epochs,log=self.log),lambda report:self.training_done(),'Menyiapkan trial dan melatih tujuh jaringan…')

    def training_done(self):self.load_metrics();self.status.set('Pelatihan selesai; nilai validasi dan uji tersedia pada tabel.')

    def load_metrics(self):
        router=ROOT/'models/commands.report.json'
        if router.is_file():
            report=json.loads(router.read_text(encoding='utf-8'));acc=report['evaluation']['test']['accuracy']
            self.router_status.set(f"Uji intent: {acc:.1%} dari {report['samples']['test']} kalimat terpisah. Ini tidak mengukur akurasi emosi EEG.")
        else:self.router_status.set('Belum dilatih.')
        path=ROOT/'models/emotion/training_report.json';self.metrics_table.delete(*self.metrics_table.get_children())
        if path.is_file():
            report=json.loads(path.read_text(encoding='utf-8'))
            self.training_status.set('Model otomatis: '+report['selected']+' (berdasarkan validasi). Akurasi uji awal rendah; lihat tabel.')
            for r in report['results']:
                self.metrics_table.insert('','end',values=(r['network'],len(r['hidden']),f"{r['validation']['macro_f1']:.3f}",f"{r['test']['macro_f1']:.3f}",f"{r['test']['accuracy']:.1%}"))
        else:self.training_status.set('Belum dilatih; sediakan dua EDF sub-02/sub-03 di data/raw lalu klik Latih.')

    def close(self):
        if self.busy:self.status.set('Tunggu proses selesai sebelum menutup.');return
        self.root.destroy()


def main():
    root=tk.Tk();App(root);root.mainloop()
