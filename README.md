# EDFLLM — model perintah kecil dan pengenal emosi EEG

Pengguna memasukkan EEG dalam EDF dan memberi perintah bahasa Indonesia. **Model pemaham perintah yang dilatih sendiri** mengenali tugas, mengambil parameter, lalu memanggil jaringan pengenal emosi. Hasil berupa garis waktu, tabel, dan keterangan terstruktur.

Tidak memakai Gemma, Ollama, API chat, atau bobot pretrained. Nama repo tetap EDFLLM; komponen bahasanya adalah **neural intent classifier kecil**, bukan LLM generatif. Tujuh jaringan EEG juga dilatih dari bobot acak dengan NumPy.

## Mulai

1. Pada instalasi baru: pasang Python 3.11+ dengan Tkinter lalu jalankan INSTALL.bat. Unduhan paket awal memerlukan internet.
2. Jalankan **JALANKAN.bat**. Model perintah dan 14 checkpoint EEG sudah tersedia pada komputer ini. Checkpoint, laporan training, dan EDF tidak diunggah ke GitHub; pada salinan repo baru jalankan latihan terlebih dahulu.
3. Tab **Data & rentang waktu**: pilih EDF A, tambahkan B untuk perbandingan. Klik **Baca file**, lalu pilih trial dari anotasi atau isi awal/durasi. Kanal berasal dari header EDF, tidak ditetapkan ke C3/C4.
4. Tab **Perintah & hasil**: ketik satu tugas dan klik **Jalankan perintah**. Tombol contoh mengisi kalimat untuk dijalankan.
5. Baca ringkasan, garis waktu, segmen, dan rincian proses. Simpan JSON/CSV/grafik bila diperlukan.

EDF lokal kini berada di `data/raw/`. Versi sebelumnya disimpan lokal di `archive/legacy-comparison-20261023/` dan riwayat Git commit `2ea5999`.

## Contoh perintah

| Kalimat | Tindakan |
|---|---|
| klasifikasikan emosi A | Prediksi kelas tiap window 4 detik pada rentang pilihan |
| cari bagian sedih di B | Prediksi lalu saring window kelas sedih yang tidak ditandai belum yakin |
| bandingkan emosi A dan B | Prediksi dua rekaman dan bandingkan proporsi durasi kelas |
| bandingkan sinyal A dan B | Hitung perbedaan statistik dan daya frekuensi |
| tampilkan informasi EDF A | Baca durasi, kanal, sampling, anotasi |
| bagaimana cara memakai aplikasi | Panduan |

Tambahkan `dengan tanh` atau nama jaringan lain; `auto` memakai skor validasi terbaik. Rentang eksplisit didukung, misalnya `detik 800 sampai 828`. Tombol **Seluruh rekaman** memilih pemrosesan bertahap seluruh file. Jeda/penilaian di luar percobaan tidak terwakili dalam latihan. Prediksi mengikuti kanal dan preprocessing model tersimpan.

## Dua jenis pembelajaran

**Pemaham perintah:** 154 kalimat dalam `corpus/commands.json`, dibagi 112 latihan, 21 validasi, 21 uji. Fitur kata/potongan karakter TF-IDF → ReLU 48 → 24 → 7 intent. Vocabulary/IDF hanya dibuat dari latihan. Parser terpisah mengambil rekaman, emosi, jaringan, interval. Dispatcher hanya memanggil alat yang diizinkan; kalimat bukan kode yang dieksekusi.

**Pengenal emosi:** sinyal → filter/resampling → window 4 detik → log daya delta/theta/alpha/beta/gamma dan log RMS tiap kanal → jaringan → tujuh skor kelas. Metadata, nama subjek, waktu dan label anotasi tidak menjadi fitur prediksi.

Kategori target: senang (joy), terinspirasi, kelembutan (tenderness), takut, jijik, sedih, netral. **Target adalah kategori stimulus**, bukan jaminan perasaan setiap window. TSV penilaian masih informasi pendamping, belum menjadi target latihan.

## Tujuh jaringan

Daftar pada modul tugas merupakan tujuh fungsi aktivasi. Masing-masing jaringan berada di file terpisah dalam `emotionlab/networks/` dan dipanggil registry.

| Jaringan | Eksperimen emosi |
|---|---|
| Sigmoid | Hidden sigmoid dan tujuh output biner one-vs-rest; BCE. Skor tidak wajib berjumlah satu. |
| ReLU | Hidden ReLU, output Softmax tujuh kelas. |
| Softmax | Eksperimen hidden kompetitif Softmax dengan Jacobian lengkap, output Softmax. Hidden Softmax bukan pilihan standar; diuji secara eksplisit. |
| Tanh | Hidden Tanh, output Softmax. |
| Leaky ReLU | Hidden dengan gradien kecil pada nilai negatif, output Softmax. |
| ELU | Hidden ELU, output Softmax. |
| Swish | Hidden Swish, output Softmax. |

Semua memakai fitur/split sama: 1 hidden `(32)` dan 3 hidden `(32,16,8)`. Pemilihan otomatis memakai macro-F1 **validasi**, bukan uji. Tidak diasumsikan setiap aktivasi memiliki kemampuan emosi khusus. Pengguna dapat memilih jaringan lewat perintah atau dropdown.

## Hasil dan batas

- Intent uji: **20/21 = 95,2%** pada corpus kecil buatan sendiri. Ini bukan akurasi emosi. Gate ketidakpastian dapat menolak sebagian kalimat yang diprediksi benar.
- Model EEG otomatis awal `tanh_baseline`: akurasi subjek uji **16,2%**, macro-F1 **0,089**. Pengenal emosi masih lemah dan belum andal.
- 84 trial menghasilkan 584 window: 196 train, 98 validasi, 290 test. Train: sub-02 stimulus 4/5; validasi: sub-02 stimulus 8; test: sub-03. Semua context dari stimulus yang sama berada di split sama. Hanya satu subjek uji.
- Skor tertinggi <0,45, margin dua kelas <0,08, atau fitur jauh dari latihan ditandai **belum yakin**. Heuristik ini belum dikalibrasi.
- Bandpass 1–45 Hz, resampling 128 Hz; belum ada ICA atau detektor artefak tervalidasi.
- Perbandingan emosi menyandingkan proporsi prediksi, bukan uji statistik atau penyelarasan stimulus otomatis.
- Evolution Algorithm ditunda. Latihan lengkap Bab 1–2 belum diklaim selesai.

Kurva dan metrik: `models/emotion/training_report.json` dan `models/commands.report.json`.

## Data dan latihan ulang

Empat JSON/TSV disertakan. Untuk salinan repo baru, unduh [EDF sub-02](https://s3.amazonaws.com/openneuro.org/ds005540/sub-02/eeg/sub-02_task-emotion_eeg.edf) dan [EDF sub-03](https://s3.amazonaws.com/openneuro.org/ds005540/sub-03/eeg/sub-03_task-emotion_eeg.edf) ke `data/raw/`.

INSTALL.bat melatih pemaham perintah setelah instalasi paket. `LATIH_PERINTAH.bat` melatih ulang model kalimat. `LATIH_EMOSI.bat` melatih semua jaringan EEG. Tombol juga tersedia di tab Model & pelatihan. Adapter latihan baru mendukung dua subjek tersebut; dataset lain perlu pemetaan trial/label baru.

Sumber pemetaan: [README resmi dataset](https://github.com/OpenNeuroDatasets/ds005540) dan [kode penulis](https://github.com/ncclab-sustech/EmoEEG-MC/blob/main/preprocess.py). Kebijakan pemotongan proyek: [docs/DATA_DAN_LABEL.md](docs/DATA_DAN_LABEL.md).

## Pembagian file

```text
main.py, cli.py               GUI dan terminal
emotionlab/commands.py        model pemaham perintah + parser
emotionlab/dispatcher.py      pemanggil alat EEG
emotionlab/dataset.py         label, trial, fitur
emotionlab/training.py        latihan dan evaluasi tujuh jaringan
emotionlab/inference.py       timeline dan perbandingan emosi
emotionlab/networks/*.py      tujuh jaringan + backprop bersama
emotionlab/edf.py, signal.py pembaca EDF dan pengolahan
emotionlab/comparison.py      pembanding sinyal
emotionlab/presentation.py    format hasil dan ekspor
emotionlab/gui.py, ui_*.py    antarmuka
corpus/commands.json          sampel kalimat dan split
models/                      checkpoint dan evaluasi lokal; tidak dipush
data/raw/                    EDF lokal dan empat pendamping
output/, archive/            hasil/cache/arsip lokal; tidak dipush
```

Jalankan UJI.bat. Uji integrasi EDF asli: `.venv\Scripts\python.exe -m tests.smoke_real`.
