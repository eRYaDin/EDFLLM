"""Sampel buatan manusia untuk domain perintah EEG, bukan percakapan umum."""
import json
from pathlib import Path

SAMPLES={
 'classify':{
 'train':['klasifikasikan emosi rekaman A','analisis emosi EEG ini','tampilkan perubahan emosi sepanjang waktu','prediksi emosi pada rekaman B','identifikasi emosi dari sinyal EEG','buat garis waktu emosi A','lihat emosi setiap segmen','tentukan kelas emosi rekaman','periksa perubahan perasaan di EEG','tampilkan prediksi per window','analisa emosi data A','emosi apa pada data ini','buat timeline emosi','cek emosi dari awal sampai akhir','kenali emosi pada segmen ini','klasifikasi emosi EEG dengan tanh'],
 'validation':['tolong klasifikasi emosi rekaman B','aku ingin melihat garis waktu emosi','analisis perasaan setiap bagian EEG'],
 'test':['buatkan prediksi emosi setiap bagian rekaman','apa emosi dalam EEG A','tunjukkan perubahan emosi dari waktu ke waktu']},
 'find':{
 'train':['cari bagian senang','temukan segmen sedih di A','kapan muncul emosi takut','tunjukkan bagian EEG yang netral','bagian mana yang senang','di detik berapa ada kesedihan','saring segmen dengan emosi jijik','pilih bagian yang terinspirasi','tampilkan hanya bagian sedih','cari waktu saat senang pada B','lokasi segmen takut dalam rekaman','temukan periode kelembutan','rekaman A bagian mana yang sedih','filter emosi netral','cari semua segmen gembira','tunjuk waktu sedih'],
 'validation':['tolong temukan bagian senang di B','kapan EEG ini menunjukkan takut','pilih hanya segmen netral'],
 'test':['di bagian mana muncul senang','tolong carikan bagian sedih rekaman A','temukan waktu yang diprediksi jijik']},
 'compare_emotion':{
 'train':['bandingkan emosi A dan B','apa perbedaan emosi kedua rekaman','bandingkan hasil klasifikasi dua EEG','apakah emosi rekaman A berbeda dari B','mana yang lebih banyak sedih A atau B','lihat perbandingan perubahan emosi','beda emosi dua subjek ini apa','bandingkan persentase senang pada dua rekaman','komparasi emosi rekaman A dengan B','seberapa berbeda distribusi emosi keduanya','bandingkan kedua data','bandingkan A dan B','beda kedua rekaman ini','apakah dua rekaman memiliki emosi sama','perbandingan emosi senang dan sedih A B','lihat beda hasil prediksi'],
 'validation':['tolong bandingkan emosi kedua file','apakah A lebih banyak senang daripada B','tampilkan perbandingan klasifikasi'],
 'test':['apa beda distribusi emosi antara rekaman A dan B','bandingkan hasil prediksi emosi keduanya','lihat apakah emosi dua EEG berbeda']},
 'compare_signal':{
 'train':['bandingkan sinyal EEG A dan B','bandingkan daya alpha kedua sinyal','lihat beda RMS dua rekaman','bandingkan spektrum EEG','apa perbedaan frekuensi sinyal A B','bandingkan amplitudo sinyal','perbandingan gelombang EEG','lihat beda daya beta pada A dan B','bandingkan bentuk sinyal keduanya','beda spektrum frekuensi dua data','komparasi sinyal A dengan B','tampilkan selisih daya pita','bandingkan energi pita frekuensi','perbandingan RMS A B','cek beda gelombang kedua rekaman','lihat dua grafik sinyal'],
 'validation':['bandingkan amplitudo kedua EEG','tampilkan perbandingan daya frekuensi','perbedaan sinyal A dengan sinyal B'],
 'test':['apa beda daya alpha antara dua rekaman','lihat perbandingan spektrum sinyal A B','bandingkan nilai RMS keduanya']},
 'metadata':{
 'train':['tampilkan informasi EDF','apa isi file ini','berapa kanal pada rekaman A','lihat sampling EEG','berapa lama durasi rekaman B','daftar kanal dalam EDF','baca metadata file','tampilkan anotasi rekaman','lihat informasi data A','berapa frekuensi sampling aslinya','jelaskan isi header EDF','tampilkan nama elektroda','cek satuan kanal','info rekaman B','apa saja kanal di sini','lihat ukuran dan durasi data'],
 'validation':['lihat metadata rekaman A','berapa jumlah kanal EEG ini','tampilkan informasi sampling'],
 'test':['beri informasi durasi file B','nama kanal apa saja yang tersedia','baca informasi header rekaman']},
 'help':{
 'train':['bagaimana cara memakai aplikasi','bantuan perintah','apa fitur yang tersedia','contoh perintah EEG','file apa yang harus dimasukkan','bagaimana melatih model','apa arti skor prediksi','tujuh jaringan itu apa','panduan penggunaan','jelaskan alur aplikasi','cara klasifikasi emosi','cara memilih rekaman','bagaimana memulai analisis','bantu saya memakai program','cara membaca hasil','model ini bisa apa'],
 'validation':['tolong beri panduan aplikasi','contoh kalimat perintah apa saja','cara memahami skor model'],
 'test':['apa saja contoh instruksi yang didukung','bagaimana menggunakan fitur EEG','apa file yang dibutuhkan program']},
 'reject':{
 'train':['berapa cuaca hari ini','buat puisi romantis','resep nasi goreng','siapa presiden','hitung dua tambah dua','nyanyikan lagu','hapus semua file','diagnosis penyakit pasien ini','apakah saya depresi','saya ingin bermain bola','ceritakan sejarah Indonesia','buat gambar kucing','beli laptop baru','apa kabar kamu','rekomendasikan obat','tulis email pekerjaan'],
 'validation':['ramalan cuaca besok','tolong buat cerita cinta','diagnosis epilepsi dari EEG'],
 'test':['beri resep masakan','siapa juara sepak bola','hapus folder data saya']}}

def main():
    root=Path(__file__).resolve().parents[1];rows=[]
    for intent,splits in SAMPLES.items():
        for split,texts in splits.items():
            rows.extend({'text':text,'intent':intent,'split':split} for text in texts)
    (root/'corpus/commands.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Corpus:',len(rows),'sampel')

if __name__=='__main__':main()
