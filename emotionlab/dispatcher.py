"""Pemanggil terstruktur: intent yang dipelajari → alat EEG; tidak ada eval/kode bebas."""
from pathlib import Path
from .commands import CommandModel
from .dataset import SETTINGS
from .edf import inspect_edf,common_voltage_channels
from .inference import predict_timeline,compare_emotions,summarize_timeline
from .comparison import compare

HELP=('Masukkan EDF A, tambahkan B untuk perbandingan. Tentukan awal dan durasi di panel data.\n'
      'Contoh: klasifikasikan emosi A; cari bagian sedih; bandingkan emosi A dan B; '
      'bandingkan sinyal A dan B; tampilkan informasi EDF.\n'
      'Boleh menyebut jaringan: sigmoid, relu, softmax, tanh, leaky_relu, elu, swish. '
      'Otomatis memilih jaringan berdasar nilai validasi saat training.\n'
      'Model perintah dilatih dari contoh kalimat. Model emosi dilatih dari sinyal dan label stimulus. '
      'Prediksi adalah hasil eksperimen awal; label stimulus tidak membuktikan perasaan setiap saat.')


def dispatch(text,records,network='auto',log=print):
    plan=CommandModel().parse(text)
    if not plan['accepted']:return {'kind':'clarify','plan':plan,'message':plan['reason']+'\n\n'+HELP}
    intent=plan['intent']
    if intent=='help':return {'kind':'help','plan':plan,'message':HELP}
    if intent=='find' and not plan['emotions']:
        return {'kind':'clarify','plan':plan,'message':'Sebutkan emosi yang ingin dicari: senang, sedih, takut, jijik, netral, terinspirasi, atau kelembutan.'}
    targets=['A','B'] if intent in ('compare_emotion','compare_signal') else (plan['records'] or ['A'])
    chosen=plan['network'] if plan['network']!='auto' else network
    inputs={}
    for key in targets:
        item=records.get(key,{})
        path=Path(item.get('file',''))
        if not path.is_file() or path.suffix.lower()!='.edf':raise ValueError('Masukkan file EDF '+key+' dahulu.')
        start,duration=plan['interval'] or [float(item.get('start',0)),float(item.get('duration',28))]
        if plan['whole_record']:start,duration=0.,inspect_edf(path)['duration_s']
        inputs[key]={'file':str(path),'start':start,'duration':duration}
    plan.update(inputs=inputs,chosen_network=chosen)
    if intent=='metadata':
        return {'kind':'metadata','plan':plan,'records':{k:inspect_edf(v['file']) for k,v in inputs.items()}}
    if intent=='compare_signal':
        a,b=inputs['A'],inputs['B']
        channels=common_voltage_channels(inspect_edf(v['file']) for v in inputs.values())
        if not channels:raise ValueError('Tidak ada kanal tegangan bersama untuk dibandingkan.')
        if abs(a['duration']-b['duration'])>1e-6:raise ValueError('Perbandingan sinyal perlu durasi A dan B sama.')
        result=compare(a['file'],b['file'],channels,a['start'],b['start'],a['duration'],SETTINGS)
        return {'kind':'signal','plan':plan,'result':result}
    timelines={key:predict_timeline(item['file'],item['start'],item['duration'],chosen,log=log) for key,item in inputs.items()}
    result={'kind':intent,'plan':plan,'timelines':timelines,'summaries':{k:summarize_timeline(v) for k,v in timelines.items()}}
    if intent=='find':
        result['matches']={k:[r for r in v['rows'] if r['display'] in plan['emotions']] for k,v in timelines.items()}
    if intent=='compare_emotion':result['comparison']=compare_emotions(timelines['A'],timelines['B'])
    return result
