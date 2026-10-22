from pathlib import Path
import math
import numpy as np
import edfio


def open_edf(path):
    record=edfio.read_edf(Path(path).resolve(),lazy_load_data=True)
    if not record.is_continuous:
        raise ValueError('EDF terputus/discontinuous belum didukung; gap tidak boleh disambung sebagai sinyal kontinu.')
    return record


def inspect_edf(path):
    path=Path(path).resolve();record=open_edf(path)
    channels=[]
    for i,signal in enumerate(record.signals):
        channels.append({'index':i,'name':signal.label.strip(),
                         'sample_rate':float(signal.sampling_frequency),
                         'samples':round(record.duration*signal.sampling_frequency),
                         'unit':signal.physical_dimension.strip(),
                         'physical_min':float(signal.physical_min),
                         'physical_max':float(signal.physical_max),
                         'prefilter':signal.prefiltering,'transducer':signal.transducer_type})
    try:start=record.startdatetime.isoformat()
    except edfio.AnonymizedDateError:start='tanggal dianonimkan'
    return {'path':str(path),'duration_s':float(record.duration),'start_time':start,
            'patient_identification':record.local_patient_identification,
            'recording':record.local_recording_identification,'channels':channels,
            'annotations':[{'onset_s':float(a.onset),'duration_s':None if a.duration is None else float(a.duration),
                            'description':a.text} for a in record.annotations]}


def unit_to_uv(unit):
    unit=unit.strip().replace('μ','u').replace('µ','u').lower()
    factors={'uv':1.,'mv':1000.,'v':1e6,'nv':.001}
    if unit not in factors:
        raise ValueError(f'Satuan {unit!r} belum didukung. Pilih kanal tegangan V/mV/uV/nV yang jelas.')
    return factors[unit]


def common_voltage_channels(infos):
    """Irisan kanal berdasarkan isi header, tanpa daftar elektroda tetap."""
    common=None
    for info in infos:
        names=[channel['name'] for channel in info['channels']];valid=[]
        for channel in info['channels']:
            try:unit_to_uv(channel['unit'])
            except ValueError:continue
            if names.count(channel['name'])==1:valid.append(channel['name'])
        common=valid if common is None else [name for name in common if name in valid]
    return common or []


def read_segment(path,channels,start,duration):
    if not channels or len(set(channels))!=len(channels):raise ValueError('Pilih kanal unik, minimal satu.')
    if not all(math.isfinite(v) for v in (start,duration)) or start<0 or duration<2:
        raise ValueError('Awal harus >=0 dan durasi minimal 2 detik, semuanya bernilai hingga.')
    record=open_edf(path);labels=[s.label.strip() for s in record.signals]
    if start+duration>record.duration+1e-6:raise ValueError(f'Segmen melewati akhir rekaman ({record.duration:g} detik).')
    result={}
    for name in channels:
        if labels.count(name)!=1:raise ValueError(f'Kanal {name!r} tidak ditemukan tepat sekali dalam EDF.')
        signal=record.signals[labels.index(name)];fs=float(signal.sampling_frequency)
        offset,size=round(start*fs),round(duration*fs)
        raw=signal.get_data_slice(offset/fs,(offset+size)/fs)
        if len(raw)!=size or not np.isfinite(raw).all():raise ValueError(f'Data kanal {name} tidak lengkap/NaN/Inf.')
        result[name]={'values_uv':raw*unit_to_uv(signal.physical_dimension),'native_fs':fs,
                      'offset_rounding_s':offset/fs-start}
    return result
