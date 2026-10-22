import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
import argparse
import json
from emotionlab.commands import train_commands
from emotionlab.training import train_all
from emotionlab.dataset import ROOT,build_manifest
from emotionlab.dispatcher import dispatch
from emotionlab.presentation import explain,export_result

def main():
    p=argparse.ArgumentParser(description='Model perintah kecil + tujuh jaringan emosi EEG')
    sub=p.add_subparsers(dest='action',required=True)
    sub.add_parser('train-commands');sub.add_parser('inspect-labels')
    t=sub.add_parser('train-emotion');t.add_argument('--epochs',type=int,default=160)
    r=sub.add_parser('run');r.add_argument('command');r.add_argument('--a');r.add_argument('--b')
    r.add_argument('--start-a',type=float,default=0);r.add_argument('--start-b',type=float,default=0)
    r.add_argument('--duration',type=float,default=28);r.add_argument('--network',default='auto');r.add_argument('--out')
    args=p.parse_args()
    if args.action=='train-commands':print(json.dumps(train_commands()['evaluation'],indent=2,ensure_ascii=False))
    elif args.action=='train-emotion':print('Model terpilih:',train_all(epochs=args.epochs)['selected'])
    elif args.action=='inspect-labels':
        rows,channels=build_manifest();print(f'{len(rows)} trial, {len(channels)} kanal');print(json.dumps(rows[:3],indent=2))
    else:
        records={'A':{'file':args.a or '', 'start':args.start_a,'duration':args.duration},'B':{'file':args.b or '', 'start':args.start_b,'duration':args.duration}}
        result=dispatch(args.command,records,args.network);print(explain(result))
        if args.out:print('Tersimpan:',export_result(result,args.out))

if __name__=='__main__':main()
