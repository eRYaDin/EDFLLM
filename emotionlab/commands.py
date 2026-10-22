"""Model neural kecil untuk intent, dilatih dari sampel; entity parser terpisah."""
import json
import re
from collections import Counter
from pathlib import Path
import numpy as np
from .dataset import ROOT
from .networks import create

ALIASES={'senang':'senang','gembira':'senang','bahagia':'senang','joy':'senang',
         'sedih':'sedih','kesedihan':'sedih','sadness':'sedih','takut':'takut','fear':'takut',
         'jijik':'jijik','disgust':'jijik','netral':'netral','neutral':'netral',
         'terinspirasi':'terinspirasi','inspiration':'terinspirasi','kelembutan':'kelembutan','tenderness':'kelembutan'}


def normalize(text):
    s=text.lower()
    for word in ALIASES:s=re.sub(r'\b'+word+r'\b','emosilabel',s)
    s=re.sub(r'\b(?:sigmoid|relu|softmax|tanh|leaky_relu|elu|swish)\b','jaringan',s)
    s=re.sub(r'\b[ab]\b','rekaman',s);s=re.sub(r'\d+(?:[.,]\d+)?','angka',s)
    return ' '.join(re.findall(r'[a-z_]+',s))


def tokens(text):
    s=normalize(text);words=s.split();items=['w:'+w for w in words]
    for word in words:
        padded='^'+word+'$'
        items.extend('c:'+padded[i:i+n] for n in (3,4) for i in range(len(padded)-n+1))
    items.extend('b:'+a+' '+b for a,b in zip(words,words[1:]))
    return Counter(items)


def vectors(texts,vocab,idf):
    x=np.zeros((len(texts),len(vocab)))
    for row,text in enumerate(texts):
        for token,count in tokens(text).items():
            if token in vocab:x[row,vocab[token]]=1+np.log(count)
    x*=idf;x/=np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-12)
    return x


def train_commands(corpus=ROOT/'corpus/commands.json',output=ROOT/'models/commands.npz'):
    rows=json.loads(Path(corpus).read_text(encoding='utf-8'));train=[r for r in rows if r['split']=='train']
    labels=sorted({r['intent'] for r in train});vocab={t:i for i,t in enumerate(sorted({t for r in train for t in tokens(r['text'])}))}
    df=np.zeros(len(vocab))
    for row in train:
        for t in tokens(row['text']):df[vocab[t]]+=1
    idf=1+np.log((1+len(train))/(1+df));sets={}
    for split in ('train','validation','test'):
        part=[r for r in rows if r['split']==split]
        sets[split]=(vectors([r['text'] for r in part],vocab,idf),np.array([labels.index(r['intent']) for r in part]))
    net=create('relu',len(vocab),len(labels),(48,24));curves=net.fit(*sets['train'],*sets['validation'],epochs=300,lr=.004,patience=35)
    meta={'schema':1,'labels':labels,'vocab':vocab,'hidden':[48,24],
          'kind':'learned intent classifier, not generative LLM','samples':{s:len(x) for s,(x,y) in sets.items()}}
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(output,metadata=np.array(json.dumps(meta)),idf=idf,train_vectors=sets['train'][0],**net.arrays())
    report={'samples':meta['samples'],'parameters':sum(p.size for p in net.weights+net.biases),'curves':curves,'evaluation':{}}
    for split in ('validation','test'):
        x,y=sets[split];pred=net.predict_scores(x).argmax(axis=1)
        report['evaluation'][split]={'accuracy':float(np.mean(pred==y)),'examples':[]}
        for row,p in zip([r for r in rows if r['split']==split],pred):
            report['evaluation'][split]['examples'].append({**row,'predicted':labels[int(p)]})
    output.with_suffix('.report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    return report


class CommandModel:
    def __init__(self,path=ROOT/'models/commands.npz'):
        if not Path(path).is_file():raise ValueError('Model perintah belum dilatih. Klik Latih pemaham perintah dahulu.')
        with np.load(path,allow_pickle=False) as p:
            self.meta=json.loads(str(p['metadata']));self.idf=p['idf'].copy();self.examples=p['train_vectors'].copy()
            self.net=create('relu',len(self.meta['vocab']),len(self.meta['labels']),tuple(self.meta['hidden']));self.net.restore(p)

    def parse(self,text):
        if not text.strip() or len(text)>500:raise ValueError('Perintah harus berisi 1–500 karakter.')
        x=vectors([text],self.meta['vocab'],self.idf);scores=self.net.predict_scores(x)[0];order=np.argsort(scores)
        similarity=float(np.max(self.examples@x[0]));intent=self.meta['labels'][int(order[-1])]
        uncertain=(scores[order[-1]]<.60 or scores[order[-1]]-scores[order[-2]]<.20 or similarity<.25)
        lower=text.lower()
        if re.search(r'\b(jangan|batal|hapus|diagnosis|diagnosa|obat)\b',lower):uncertain=True
        found=[key for key in ALIASES if re.search(r'\b'+key+r'\b',lower)]
        emotions=list(dict.fromkeys(ALIASES[word] for word in found))
        records=list(dict.fromkeys(s.upper() for s in re.findall(r'\b([ab])\b',lower)))
        networks=re.findall(r'\b(sigmoid|relu|softmax|tanh|leaky_relu|elu|swish)\b',lower.replace('leaky relu','leaky_relu'))
        if len(set(networks))>1:uncertain=True
        # Explicit bounded interval only; other numeric wording is rejected, not guessed.
        interval=re.search(r'(?:detik\s+)?(\d+(?:\.\d+)?)\s*(?:sampai|hingga|[-–])\s*(\d+(?:\.\d+)?)\s*(?:detik|s)?',lower)
        span=None
        if interval:
            begin,end=map(float,interval.groups())
            if end<=begin:raise ValueError('Akhir waktu harus sesudah awal.')
            span=[begin,end-begin]
        elif re.search(r'\d',lower):uncertain=True
        whole=bool(re.search(r'\b(seluruh rekaman|seluruh file|sepanjang rekaman|dari awal sampai akhir)\b',lower))
        if whole and span:uncertain=True
        return {'intent':intent,'accepted':not uncertain and intent!='reject','score':float(scores[order[-1]]),
                'similarity':similarity,'emotions':emotions,'records':records,'network':networks[0] if networks else 'auto',
                'interval':span,'whole_record':whole,'text':text,'reason':'Perintah belum dikenali dengan cukup yakin; pilih contoh atau tulis satu tugas EEG.' if uncertain or intent=='reject' else ''}
