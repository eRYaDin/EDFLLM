import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
import edfio
from emotionlab.networks import NETWORKS,create
from emotionlab.dataset import trials_from_info,SEQUENCES,LABELS,features,SETTINGS
from emotionlab.commands import CommandModel
from emotionlab.dispatcher import dispatch
from emotionlab.training import load_model
from emotionlab.inference import predict_timeline,compare_emotions,summarize_timeline


class Tests(unittest.TestCase):
    def test_all_seven_gradients_and_different_activations(self):
        x=np.array([[.2,-.3],[.5,.1],[-.1,.4]]);y=np.array([0,1,2])
        for name in NETWORKS:
            net=create(name,2,3,(4,3));loss,gw,gb=net.loss_gradients(x,y)
            for layer,pos in [(0,(0,0)),(1,(1,1)),(2,(0,2))]:
                w=net.weights[layer];old=w[pos];eps=1e-6
                w[pos]=old+eps;plus=net.loss_gradients(x,y)[0]
                w[pos]=old-eps;minus=net.loss_gradients(x,y)[0];w[pos]=old
                self.assertAlmostEqual(gw[layer][pos],(plus-minus)/(2*eps),places=6,msg=name)
            self.assertTrue(np.isfinite(loss));self.assertEqual(net.predict_scores(x).shape,(3,3))

    def test_label_mapping_counts_boundaries_and_split(self):
        annotations=[];t=10
        for code in (33,32):
            for i in range(21):
                annotations.extend([{'onset_s':t,'description':f'TypeID: {code}'},
                                    {'onset_s':t+60,'description':'TypeID: 31'}]);t+=90
        info={'path':'example.edf','duration_s':4000,'annotations':annotations}
        rows=trials_from_info(info,'sub-02')
        self.assertEqual(len(rows),42);self.assertEqual(rows[0]['label'],'takut')
        self.assertEqual(rows[0]['split'],'validation');self.assertEqual(rows[1]['split'],'train')
        self.assertTrue(all(r['start_s']>=r['onset_s'] and r['start_s']+r['duration_s']<r['boundary_s'] for r in rows))
        self.assertFalse({r['stimulus'] for r in rows if r['split']=='train'} & {r['stimulus'] for r in rows if r['split']=='validation'})
        info['annotations']=annotations[:-2]
        with self.assertRaises(ValueError):trials_from_info(info,'sub-02')

    def test_features_keep_window_order_and_dynamic_channels(self):
        t=np.arange(8*128)/128
        a=np.r_[np.sin(2*np.pi*10*t[:512]),np.sin(2*np.pi*20*t[:512])]
        x,names=features(np.array([a,a*.5]),['AF3','Oz'])
        self.assertEqual(x.shape,(2,12));self.assertTrue(all(name.startswith(('AF3:','Oz:')) for name in names))
        self.assertGreater(x[0,names.index('AF3:log_power_alpha')],x[1,names.index('AF3:log_power_alpha')])

    def test_learned_router_and_safe_dispatch(self):
        model=CommandModel()
        for text,intent in [('cari bagian sedih','find'),('klasifikasikan emosi A','classify'),
                            ('bandingkan emosi A dan B','compare_emotion'),('bandingkan sinyal A dan B','compare_signal')]:
            plan=model.parse(text);self.assertTrue(plan['accepted'],text);self.assertEqual(plan['intent'],intent)
        self.assertFalse(model.parse('hapus folder data saya')['accepted'])
        self.assertFalse(model.parse('jangan klasifikasikan emosi A')['accepted'])
        result=dispatch('hapus folder data saya',{})
        self.assertEqual(result['kind'],'clarify')
        with self.assertRaises(ValueError):dispatch('klasifikasikan emosi A',{})
        slots=model.parse('cari bagian sedih di B detik 800 sampai 828 dengan tanh')
        self.assertEqual(slots['interval'],[800.,28.]);self.assertEqual(slots['records'],['B']);self.assertEqual(slots['emotions'],['sedih'])

    def test_predictions_ignore_annotation_labels_and_serialization(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1],prefix='test-') as temp:
            folder=Path(temp);t=np.arange(12*128)/128
            samples=[np.sin(2*np.pi*10*t)*10,np.cos(2*np.pi*20*t)*5]
            for name,label in [('a','joy'),('b','sad')]:
                signals=[edfio.EdfSignal(x,128,label=ch,physical_dimension='uV') for x,ch in zip(samples,['AF3','Oz'])]
                edfio.Edf(signals,annotations=[edfio.EdfAnnotation(0,12,label)]).write(folder/(name+'.edf'))
            x,names=features(np.array(samples),['AF3','Oz'])
            net=create('tanh',len(names),len(LABELS),(4,));meta={'schema':2,'id':'test','network':'tanh','hidden':[4],
                'features':names,'channels':['AF3','Oz'],'labels':LABELS,'settings':SETTINGS.to_dict(),
                'label_source':'test','test':{'accuracy':0,'macro_f1':0},'train_subjects':[]}
            np.savez_compressed(folder/'test.npz',metadata=np.array(json.dumps(meta)),mean=np.zeros(len(names)),std=np.ones(len(names)),**net.arrays())
            (folder/'training_report.json').write_text(json.dumps({'selected':'test'}))
            restored,*_=load_model(folder/'test.npz');np.testing.assert_allclose(net.predict_scores(x),restored.predict_scores(x))
            a=predict_timeline(folder/'a.edf',0,10,model_dir=folder,log=lambda s:None)
            b=predict_timeline(folder/'b.edf',0,10,model_dir=folder,log=lambda s:None)
            self.assertEqual(a['rows'],b['rows']);self.assertEqual(a['ignored_tail_s'],2)
            self.assertEqual([r['start_s'] for r in a['rows']],[0,4])
            self.assertEqual(sum(v['seconds'] for v in summarize_timeline(a).values()),8)
            self.assertTrue(all(r['B_minus_A_pp']==0 for r in compare_emotions(a,b)))
            with self.assertRaises(ValueError):predict_timeline(folder/'a.edf',10,4,model_dir=folder)


if __name__=='__main__':unittest.main()
