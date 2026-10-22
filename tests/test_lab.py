import csv
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
import edfio
from eeglab.demo import make_demo,write_signal
from eeglab.edf_reader import inspect_edf,read_segment,unit_to_uv,common_voltage_channels
from eeglab.preprocessing import prepare,Settings
from eeglab.comparison import compare,export
from eeglab.features import describe
from eeglab.model_registry import create_model,MODELS
from eeglab.training import load_manifest,group_split,train_manifest,predict_file,resolve_training_channels

ROOT=Path(__file__).resolve().parent.parent


class LabTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory(dir=ROOT,prefix='test-')
        cls.folder=make_demo(Path(cls.temp.name)/'demo');cls.a=cls.folder/'banding_A.edf';cls.b=cls.folder/'banding_B.edf'

    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()

    def test_real_edf_headers_and_annotations(self):
        info=inspect_edf(self.a)
        self.assertEqual(info['duration_s'],16)
        self.assertEqual([c['name'] for c in info['channels']],['C3','C4'])
        self.assertEqual(info['channels'][0]['sample_rate'],160)
        self.assertEqual(info['annotations'][0]['description'],'SYNTHETIC_NOT_HUMAN_EEG')

    def test_unit_conversion(self):
        other=self.folder/'same_mV.edf';write_signal(other,unit='mV')
        x=read_segment(self.a,['C3'],0,4)['C3']['values_uv']
        y=read_segment(other,['C3'],0,4)['C3']['values_uv']
        np.testing.assert_allclose(x,y,atol=.02)
        self.assertEqual(unit_to_uv('µV'),1)
        with self.assertRaises(ValueError):unit_to_uv('')

    def test_segment_and_channel_errors(self):
        with self.assertRaises(ValueError):read_segment(self.a,['missing'],0,4)
        with self.assertRaises(ValueError):read_segment(self.a,['C3'],15,4)
        with self.assertRaises(ValueError):read_segment(self.a,['C3'],float('nan'),4)
        with self.assertRaises(ValueError):prepare(self.a,['C3'],0,4,Settings(fs=200))

    def test_comparison_psd_direction_and_export(self):
        result=compare(self.a,self.b,['C3','C4'],0,0,10,Settings())
        self.assertEqual(result['signals_A'].shape,(2,1280))
        rows={r['metric']:r for r in result['report']['comparison'] if r['channel']=='C3'}
        self.assertGreater(rows['power_alpha_uv2']['A'],rows['power_alpha_uv2']['B']*5)
        self.assertGreater(rows['power_beta_uv2']['B'],rows['power_beta_uv2']['A']*5)
        path=export(result,Path(self.temp.name)/'export')
        for name in ('laporan.json','perbandingan.csv','grafik.png'):self.assertGreater((path/name).stat().st_size,100)

    def test_unavailable_band_not_fabricated(self):
        x,_=prepare(self.a,['C3'],0,4,Settings(high=30))
        values,_,_=describe(x[0],Settings(high=30))
        self.assertIsNone(values['power_gamma_uv2'])

    def test_group_splits_disjoint_and_file_leak_rejected(self):
        rows=load_manifest(self.folder/'manifest.csv');splits=group_split(rows)
        sets=[{rows[int(i)]['group'] for i in split} for split in splits]
        self.assertFalse(sets[0]&sets[1]);self.assertFalse(sets[0]&sets[2]);self.assertFalse(sets[1]&sets[2])
        bad=Path(self.temp.name)/'bad.csv'
        with bad.open('w',newline='') as file:
            writer=csv.DictWriter(file,fieldnames=['file','label','group','start_s','duration_s']);writer.writeheader()
            for group in ('one','two'):writer.writerow({'file':str(self.a),'label':'x','group':group,'start_s':0,'duration_s':4})
        with self.assertRaisesRegex(ValueError,'dua group'):load_manifest(bad)

    def test_manifest_empty_folder_and_wrong_file_rejected(self):
        for value,message in [('', 'belum dipilih'),('   ','belum dipilih'),
                              (self.folder,'folder'),(self.a,'harus file')]:
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError,message):load_manifest(value)
        behavior=Path(self.temp.name)/'behavior.tsv'
        behavior.write_text('trial_number\tvideo_name\tscore_1\n1\t35\t2.3\n',encoding='utf-8')
        with self.assertRaisesRegex(ValueError,'belum berupa manifest'):load_manifest(behavior)

    def test_activations_gradients_all_seven(self):
        rng=np.random.default_rng(4);x=rng.normal(size=(4,3));y=np.array([0,1,1,0])
        for name in MODELS:
            with self.subTest(name=name):
                net=create_model(name,3,2,hidden=(4,),seed=2)
                loss,gw,gb=net.loss_gradients(x,y)
                for layer,index in [(0,(1,2)),(1,(0,0))]:
                    old=net.weights[layer][index];epsilon=1e-5
                    net.weights[layer][index]=old+epsilon;plus=net.loss_gradients(x,y)[0]
                    net.weights[layer][index]=old-epsilon;minus=net.loss_gradients(x,y)[0]
                    net.weights[layer][index]=old
                    self.assertAlmostEqual((plus-minus)/(2*epsilon),gw[layer][index],places=5)
                p=net.predict_proba(x);np.testing.assert_allclose(p.sum(axis=1),1,atol=1e-8)
                self.assertTrue(np.isfinite(loss))
        with self.assertRaises(ValueError):create_model('sigmoid',3,7)

    def test_training_channels_follow_manifest(self):
        rows=load_manifest(self.folder/'manifest.csv')
        self.assertEqual(resolve_training_channels(rows,'auto'),['C3','C4'])
        self.assertEqual(resolve_training_channels(rows,'C4'),['C4'])
        with self.assertRaisesRegex(ValueError,'Kanal yang tersedia: C3, C4'):
            resolve_training_channels(rows,['Fp1'])

    def test_arbitrary_edf_channels_not_fixed_to_demo(self):
        file=Path(self.temp.name)/'arbitrary.edf'
        signal=np.sin(2*np.pi*10*np.arange(1280)/128)
        record=edfio.Edf([edfio.EdfSignal(signal,128,label=name,physical_dimension=unit)
                          for name,unit in [('Fz','uV'),('Pz','uV'),('Aux','BPM')]])
        record.write(file)
        self.assertEqual(common_voltage_channels([inspect_edf(file)]),['Fz','Pz'])
        self.assertEqual(resolve_training_channels([{'file':str(file)}],'auto'),['Fz','Pz'])

    def test_training_all_models_and_reload(self):
        out=Path(self.temp.name)/'models'
        report=train_manifest(self.folder/'manifest.csv',['C3','C4'],Settings(),out,epochs=40,log=lambda message:None)
        self.assertEqual(len(report['results']),7)
        for row in report['results']:
            self.assertEqual(row['status'],'trained')
            prediction=predict_file(out/(row['model']+'.npz'),self.a,0,10)
            self.assertEqual(prediction['windows'],5)
            self.assertAlmostEqual(sum(prediction['mean_probabilities'].values()),1)
            self.assertEqual(set(prediction['labels']),{'alpha_kuat','beta_kuat'})
        saved=json.loads((out/'training_report.json').read_text())
        self.assertEqual(len(saved['split_groups']['test']),2)


if __name__=='__main__':unittest.main()
