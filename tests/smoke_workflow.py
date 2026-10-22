import tkinter as tk
from pathlib import Path
from eeglab.gui import App
from eeglab.workflow import inspect_pair,suggested_sidecars
from eeglab.comparison import compare
from eeglab.preprocessing import Settings
from eeglab.llm import context
root=tk.Tk();root.withdraw();app=App(root);root.update()
assert str(app.nav['assistant']['state'])=='disabled'
paths={k:str(Path('data/edf')/f'sub-0{n}_task-emotion_eeg.edf') for k,n in [('A',2),('B',3)]}
pair=inspect_pair(paths,{k:suggested_sidecars(p) for k,p in paths.items()})
assert len(pair['channels'])==64
app.inspected(pair);root.update();assert len(app.selected())==64
result=compare(paths['A'],paths['B'],pair['channels'],0,0,10,Settings())
app.analyzed(result);root.update();assert len(app.table.get_children())==9
assert app.current_page=='results'
data=context(result['report'],pair['channels'][0]);assert len(data['comparison'])==9
assert 'file_A' not in data
app.paths['A'].set(paths['A']);assert app.result is None and app.pair is None
root.destroy()
print('GUI workflow passed: real EDF, 64 channels, 576 metrics, result reset, selected-channel LLM context.')
