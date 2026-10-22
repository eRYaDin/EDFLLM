import csv


def read_events(path):
    """Baca tabel TSV pendamping; tidak menebak arti trigger sebagai emosi."""
    with open(path,encoding='utf-8-sig',newline='') as f:
        reader=csv.DictReader(f,delimiter='\t')
        columns=reader.fieldnames or []
        if not columns:raise ValueError('TSV tidak memiliki header.')
        rows=list(reader)
    return {'path':str(path),'columns':columns,'rows':rows,
            'note':'Data tabel asli. Pemetaan trigger/label dan keselarasan waktunya harus mengikuti dokumentasi dataset.'}
