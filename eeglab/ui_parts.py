"""Komponen desktop kecil yang dipakai oleh halaman alur."""
from tkinter import ttk
from tkinter.scrolledtext import ScrolledText


def paragraph(parent,text,width=880):
    widget=ttk.Label(parent,text=text,wraplength=width,justify='left',style='Body.TLabel')
    widget.pack(anchor='w',fill='x',pady=(0,8));return widget


def section(parent,title):
    frame=ttk.LabelFrame(parent,text=title,padding=12)
    frame.pack(fill='x',pady=(0,10));return frame


def text_area(parent,height=8):
    widget=ScrolledText(parent,wrap='word',font=('Segoe UI',10),height=height,
                        bg='#ffffff',fg='#243340',relief='flat',padx=12,pady=10)
    widget.pack(fill='both',expand=True);widget.configure(state='disabled');return widget


def set_text(widget,text):
    widget.configure(state='normal');widget.delete('1.0','end');widget.insert('end',text);widget.configure(state='disabled')


def table(parent,columns,widths=None,height=8):
    frame=ttk.Frame(parent);frame.pack(fill='both',expand=True)
    widget=ttk.Treeview(frame,columns=list(columns),show='headings',height=height)
    for i,(key,label) in enumerate(columns.items()):
        widget.heading(key,text=label);widget.column(key,width=(widths or [160]*len(columns))[i],minwidth=65)
    scroll=ttk.Scrollbar(frame,orient='vertical',command=widget.yview)
    widget.configure(yscrollcommand=scroll.set);scroll.pack(side='right',fill='y');widget.pack(fill='both',expand=True)
    return widget
