import argparse
import json
from pathlib import Path
from eeglab.preprocessing import Settings
from eeglab.comparison import compare,export
from eeglab.summary import summarize
from eeglab.training import train_manifest,predict_file
from eeglab.edf_reader import inspect_edf
from eeglab.model_registry import MODELS


def main():
    parser=argparse.ArgumentParser(description='EDF Lab — GUI: python main.py')
    sub=parser.add_subparsers(dest='action',required=True)
    inspect=sub.add_parser('info');inspect.add_argument('file')
    comp=sub.add_parser('compare');comp.add_argument('a');comp.add_argument('b')
    comp.add_argument('--start-a',type=float,default=0);comp.add_argument('--start-b',type=float,default=0)
    comp.add_argument('--duration',type=float,default=10);comp.add_argument('--out',default='output/perbandingan')
    train=sub.add_parser('train');train.add_argument('manifest');train.add_argument('--model',choices=['semua',*MODELS],default='semua')
    train.add_argument('--epochs',type=int,default=120);train.add_argument('--depth',choices=['baseline','deep','both'],default='both')
    train.add_argument('--out',default='output/models')
    for command in (comp,train):
        command.add_argument('--channels',required=True,help='Nama persis, dipisahkan koma')
        command.add_argument('--fs',type=float,default=128);command.add_argument('--low',type=float,default=1);command.add_argument('--high',type=float,default=45)
    pred=sub.add_parser('predict');pred.add_argument('model');pred.add_argument('edf')
    pred.add_argument('--start',type=float,default=0);pred.add_argument('--duration',type=float,default=10)
    args=parser.parse_args()
    if args.action=='info':print(json.dumps(inspect_edf(args.file),ensure_ascii=False,indent=2));return
    if args.action=='predict':print(json.dumps(predict_file(args.model,args.edf,args.start,args.duration),ensure_ascii=False,indent=2));return
    settings=Settings(fs=args.fs,low=args.low,high=args.high);channels=[c.strip() for c in args.channels.split(',')]
    if args.action=='compare':
        result=compare(args.a,args.b,channels,args.start_a,args.start_b,args.duration,settings)
        print(summarize(result['report']));print('Laporan:',export(result,args.out))
    else:
        for depth,hidden in [('baseline',(32,)),('deep',(32,16,8))]:
            if args.depth not in (depth,'both'):continue
            report=train_manifest(args.manifest,channels,settings,Path(args.out)/depth,
                                   list(MODELS) if args.model=='semua' else [args.model],args.epochs,hidden)
            print(depth,[(r['model'],r.get('metrics',r.get('reason'))) for r in report['results']])


if __name__=='__main__':main()
