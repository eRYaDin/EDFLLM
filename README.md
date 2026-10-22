# EEG EDF Lab

Aplikasi desktop Python untuk membaca dua EDF, menjelaskan isi rekamannya, membandingkan sinyal, serta melatih dan menjalankan tujuh eksperimen jaringan. Seluruh kode berada di folder ini; tidak mengimpor atau mengubah aplikasi Rin maupun Modul Perasaan lain.

## Mulai

Klik **JALANKAN.bat** setelah menjalankan INSTALL.bat pada instalasi baru.

1. **Masukkan data:** pilih EDF A dan EDF B. Boleh file yang sama untuk membandingkan dua waktu. JSON/TSV bernama cocok ditemukan otomatis dan bersifat opsional.
2. **Pilih segmen:** tentukan awal A/B dan durasi. Semua kanal tegangan bersama mengikuti header EDF; bisa dibatasi manual. Periksa sampling dan filter, lalu hitung perbandingan.
3. **Lihat hasil:** pilih kanal; baca penjelasan RMS/daya pita, grafik, dan tabel dengan satuan. Simpan laporan bila diperlukan.
4. **Tanya AI:** aktifkan Ollama, masukkan nama model lokal, lalu tanyakan hasil. LLM menerima angka hasil hitung kanal pilihan dan menjelaskannya. Percakapan direset ketika kanal atau perbandingan berubah.

**Laboratorium model** terpisah untuk training dan prediksi. Membandingkan EDF tidak memerlukan manifest atau model terlatih. CSV hasil perbandingan dan beh.tsv bukan manifest training. Lihat [panduan alur](PANDUAN_ALUR.md).

Untuk instalasi ulang/perangkat lain: pasang Python 3.11+ dengan Tkinter, lalu jalankan INSTALL.bat. Internet diperlukan untuk paket awal, bukan untuk pembandingan EDF. Environment memakai edfio agar tidak membutuhkan build C pembaca EDF pada Python 3.14.

## Yang sudah bisa dipakai

- Membaca EDF/EDF+ kontinu: informasi pasien/rekaman yang tersedia, waktu/durasi, nama kanal, sampling per kanal, satuan, rentang skala fisik, prefilter, transducer, dan anotasi.
- Membaca TSV pendamping sebagai tabel melalui tombol masing-masing rekaman. Arti trigger tidak ditebak menjadi emosi. Tampilan TSV dibatasi 200 baris.
- Mencocokkan nama kanal persis, tanpa menganggap C3 sama dengan C3-A2 atau EEG C3-REF. Nama duplikat tidak dipilih otomatis.
- Mengubah satuan V/mV/uV/nV menjadi uV. Kanal tanpa satuan tegangan yang dikenal ditolak untuk analisis.
- Resampling dengan filter anti-alias, band-pass Butterworth orde 4 zero-phase, dan parameter sama untuk kedua rekaman. Target sampling tidak boleh melebihi sampling asli kanal terpilih.
- Menghitung mean, std, RMS, peak-to-peak dan daya pita delta 1–4, theta 4–8, alpha 8–13, beta 13–30, gamma 30–45 Hz. Daya pita hanya dilaporkan jika seluruh pita berada dalam rentang filter; jika tidak, ditandai tidak tersedia.
- Membandingkan nilai A, B dan B-A. Amplitudo uV, PSD uV²/Hz, daya pita uV². Definisi pita ini konfigurasi proyek, bukan otomatis mengikuti setiap dataset.
- Menampilkan indikator dasar kualitas dari data asli: RMS, rentang, dan proporsi sampel berurutan yang sama. Ini belum detektor artefak tervalidasi.
- Training, evaluasi, penyimpanan, dan pemuatan tujuh model; baseline satu hidden layer versus tiga hidden layer.
- Penjelasan opsional melalui Ollama lokal, terpisah dari hitungan. Tidak mengirim nama/identitas subjek atau seluruh sampel ke prompt LLM. Ollama harus sudah aktif dan memiliki model yang diminta.

## Pembagian file

```text
main.py                         titik masuk GUI
cli.py                          pemanggil lewat terminal
buat_demo.py                    pembuat EDF sintetis
eeglab/
  gui.py                        antarmuka dan worker
  edf_reader.py                 header, anotasi, sampel, satuan
  events.py                     tabel TSV pendamping
  preprocessing.py              resampling dan filter
  features.py                   statistik, PSD, fitur window
  comparison.py                 pembanding dan ekspor
  summary.py                    ringkasan deterministik
  llm.py                        penjelasan Ollama opsional
  model_registry.py             pemanggil tujuh model
  training.py                   dataset, split, training, inferensi
  models/
    base.py                     mesin dense layer, backprop, Adam
    sigmoid.py                  model 1
    relu.py                     model 2
    softmax.py                  model 3
    tanh.py                     model 4
    leaky_relu.py               model 5
    elu.py                      model 6
    swish.py                    model 7
demo/                           EDF sintetis + manifest
output/                         hasil perbandingan dan model
tests/test_lab.py               pengujian
```

Ketujuh model tidak disatukan dalam satu file. Base memuat operasi bersama, registry memilih kelas, dan training memanggil kelas itu. Evolusi ditunda sesuai pembahasan; tidak disertakan. Latihan lengkap Bab 1–2 belum menjadi bagian folder aplikasi ini.

## Tujuh eksperimen dan layer

| Model | Hidden layer | Output | Tugas |
|---|---|---|---|
| Sigmoid | Sigmoid | Satu Sigmoid | Klasifikasi tepat dua kelas, binary cross-entropy |
| ReLU | ReLU | Softmax | Klasifikasi dua atau lebih kelas |
| Softmax | ReLU | Softmax | Baseline peran Softmax sebagai output |
| Tanh | Tanh | Softmax | Alternatif aktivasi hidden pada tugas sama |
| Leaky ReLU | Leaky ReLU, slope 0.01 | Softmax | Alternatif aktivasi hidden pada tugas sama |
| ELU | ELU, alpha 1 | Softmax | Alternatif aktivasi hidden pada tugas sama |
| Swish | z × sigmoid(z) | Softmax | Alternatif aktivasi hidden pada tugas sama |

ReLU dan eksperimen Softmax memakai arsitektur sama, sehingga hasilnya bisa identik dengan seed yang sama. Tujuh ini bukan tujuh algoritma yang independen atau tujuh keahlian EEG eksklusif. Untuk label lebih dari dua kelas, eksperimen Sigmoid dilewati dan alasannya dicatat; buat manifest biner terpisah jika ingin melatihnya. Tidak menggabungkan kelas netral ke positif/negatif secara otomatis.

Baseline: `jumlah_fitur → 32 → output`. Versi lebih dalam: `jumlah_fitur → 32 → 16 → 8 → output`. Layer input/output tidak dihitung sebagai hidden layer. Parameter diinisialisasi dengan seed 42. Training menggunakan mini-batch 32, Adam lr 0.001, gradient clipping norm 5, early stopping validation loss (patience 15), dan pemulihan bobot validation terbaik. Kedalaman lebih besar belum tentu lebih baik.

Input model adalah fitur per window non-overlap 2 detik, disusun menurut urutan kanal dan nama fitur. Sisa segmen yang tidak penuh satu window dibuang. Normalisasi fitur di-fit pada train saja dan disimpan di .npz. Output model berupa label/skor, bukan interpretasi emosi tanpa pelatihan yang sesuai.

## Training dengan data sendiri

Siapkan manifest CSV atau TSV dengan kolom wajib berikut:

```csv
file,label,group,start_s,duration_s
sub01.edf,joy,sub01,120,30
sub01.edf,sadness,sub01,180,30
sub02.edf,joy,sub02,100,30
```

Contoh di atas hanya format, bukan dataset yang cukup untuk training. Path relatif mengacu ke folder manifest, bukan folder program. `group` harus mewakili unit independen yang ingin diuji, biasanya subjek. Semua segmen dari EDF yang sama wajib satu group. Untuk satu subjek dengan banyak sesi, gunakan ID subjek sama jika menilai generalisasi antar-subjek.

Minimal lima group, minimal dua kelas, dan semua kelas harus tersedia pada setiap split. Pembagian train/validation/test sekitar 60/20/20 berdasarkan group, dilakukan sebelum pembuatan window. Satu file tidak boleh masuk dua group; segmen tumpang tindih/duplikat dalam file sama ditolak. Aplikasi mencoba 100 seed split berurutan mulai 42 untuk memperoleh cakupan kelas; daftar group aktual disimpan. Salinan file dengan nama berbeda tidak otomatis terdeteksi sebagai duplikat: pengguna tetap perlu mengaudit dataset.

Di tab model: pilih manifest, pilih kanal, model atau semua, dan epoch; tekan **Latih & bandingkan 1 vs 3 hidden layer**. Model disimpan di `output/models/<waktu>/baseline/` dan `deep/`, satu .npz per model. Laporan menyimpan kurva loss, accuracy, balanced accuracy, macro F1, confusion matrix, daftar label, settings dan group split. Metrik dihitung per window, bukan per subjek/trial.

Kanal training sekarang diisi pada kolom tersendiri di tab model, terpisah dari daftar kanal perbandingan A/B. Default `auto` mengambil semua kanal tegangan bernama sama yang tersedia pada seluruh EDF dalam manifest. Untuk membatasi kanal, isi nama dipisahkan koma, misalnya `C3,C4`. Karena itu, manifest demo tetap bisa dilatih saat rekaman A/B asli sedang memilih Fp1. Nama kanal dari perbandingan tidak dipakai diam-diam untuk training.

Untuk memprediksi EDF baru, klik **Muat model .npz → prediksi A dan B**. Kanal, preprocessing, urutan fitur, label, dan normalisasi mengikuti model tersimpan. Waktu/durasi mengikuti input GUI. Rekaman baru harus memakai protokol, referensi, dan distribusi yang sesuai pelatihan; model tidak menjamin generalisasi ke sembarang EDF.

### Untuk EmoEEG-MC

EDF, trigger, file events, dan urutan trial harus dicocokkan dari dokumentasi dataset. Aplikasi ini tidak otomatis memetakan TypeID menjadi tujuh emosi, tidak mengurutkan trial, dan tidak menganggap kolom rating sebagai label. TSV bisa dibuka untuk inspeksi, lalu manifest disusun setelah pemetaan benar. Label hilang atau trial hilang tidak boleh dikarang. Dukungan .npy DE/PSD dari derivatives belum diimplementasikan.

## Contoh terminal

```powershell
.venv\Scripts\python.exe cli.py info demo\banding_A.edf
.venv\Scripts\python.exe cli.py compare demo\banding_A.edf demo\banding_B.edf --channels C3,C4 --duration 10 --out output\contoh
.venv\Scripts\python.exe cli.py train demo\manifest.csv --channels C3,C4 --model semua --depth both --epochs 120 --out output\models\demo
.venv\Scripts\python.exe cli.py predict output\models\demo\deep\relu.npz demo\banding_A.edf --duration 10
```

## Batas perbandingan

- Perbandingan statistik tidak membutuhkan ML. Tabel hitungan selalu dapat digunakan tanpa model terlatih atau Ollama.
- Segmen A/B dipilih manual. Anotasi ditampilkan, tetapi penyelarasan otomatis berdasarkan stimulus belum dibuat.
- Belum ada re-referencing, ICA, interpolasi kanal buruk, atau pemeriksaan kesetaraan montase otomatis.
- Filter memakai konteks tambahan hingga 2 detik sebelum/sesudah bila tersedia, lalu dipotong ke segmen pilihan. Tepi file masih bisa bertransien; sampling sama tidak membuktikan kondisi eksperimennya sama.
- Tidak melaporkan beda signifikan secara statistik, diagnosis, atau kepastian emosi dari perbedaan power.
- EDF terputus ditolak. Pembaca dapat menolak EDF nonstandar. Tidak ada dukungan BDF/GDF di versi ini.
- Data dan metadata ekspor mengikuti file yang dipilih, termasuk identitas yang tersedia di header. Pengguna menentukan lokasi ekspor.
- Worker menjaga GUI responsif, tetapi belum ada pembatalan training/inferensi. Tunggu selesai sebelum menutup.
- Modul LLM hanya menjelaskan hasil hitungan atas permintaan. Routing alat mandiri dari percakapan belum dibuat; pemanggil model saat ini registry dan kontrol GUI/CLI.

## Demo dan pengujian

Demo dibuat dari sinus 10 Hz/20 Hz + noise dengan label `alpha_kuat`/`beta_kuat`. Ada dua EDF pembanding dengan sampling dan satuan berbeda, serta 10 group sintetis untuk latihan. Bukan data manusia dan bukan model pengenal emosi.

Jalankan UJI.bat. Tes memeriksa pembacaan EDF/annotasi, konversi satuan, penolakan input, perbandingan PSD, ekspor, split group, gradien tujuh model, training, dan pemuatan model. Bukti uji aktual dicatat di KEMAJUAN.md.

## Sumber teknis

- [edfio read_edf](https://edfio.readthedocs.io/en/stable/generated/edfio.read_edf.html)
- [SciPy Welch](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.welch.html)
- [SciPy resample_poly](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.resample_poly.html)
- [GroupShuffleSplit](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupShuffleSplit.html)
