# Panduan alur EEG / EDF Lab

Aplikasi membandingkan angka dan spektrum dua segmen EEG. LLM menjelaskan hasil hitungan; tujuh jaringan merupakan eksperimen klasifikasi terpisah yang perlu dilatih dari data berlabel.

| Halaman | File yang diperlukan | Hasil yang diterima |
|---|---|---|
| 1. Masukkan data | EDF A dan B | Durasi, kanal, sampling, anotasi |
| Pendamping opsional | JSON pengaturan; TSV kejadian/penilaian | Rincian sumber, tidak otomatis menjadi label |
| 2. Pilih segmen | Tidak ada file tambahan | Awal/durasi, kanal, filter untuk perhitungan |
| 3. Lihat hasil | Tidak ada file tambahan | Penjelasan, grafik, angka A/B dan selisih beserta satuan |
| 4. Tanya AI | Ollama aktif dan model lokal tersedia | Penjelasan angka kanal pilihan, bukan klasifikasi baru |
| Laboratorium: latih | Manifest CSV/TSV dengan file,label,group,start_s,duration_s dan EDF terkait | Perbandingan 1 vs 3 hidden layer, metrik uji, model .npz |
| Laboratorium: prediksi | Model .npz hasil training dan EDF sesuai kanal model | Skor label yang memang dilatih |

## Data dalam repository

Empat file pendamping sub-02/sub-03 tersedia di data/edf. Sumber dataset: OpenNeuro ds005540. File EDF tidak disertakan; unduh sendiri dan tempatkan di folder yang sama. JSON dan TSV saja tidak berisi gelombang EEG dan tidak cukup untuk menjalankan analisis sinyal.

- [EDF sub-02](https://s3.amazonaws.com/openneuro.org/ds005540/sub-02/eeg/sub-02_task-emotion_eeg.edf)
- [EDF sub-03](https://s3.amazonaws.com/openneuro.org/ds005540/sub-03/eeg/sub-03_task-emotion_eeg.edf)

Pilih file EDF melalui tombol di halaman 1. File pendamping ditemukan berdasarkan nama. Nama kanal mengikuti EDF; tidak diwajibkan C3/C4. Anotasi boleh membantu memilih waktu, tetapi aplikasi tidak menebak makna trigger menjadi emosi.

## Membaca hasil

RMS: amplitudo efektif dalam µV. Daya pita: integral PSD pada rentang frekuensi dalam µV². Grafik PSD memakai µV²/Hz. Selisih B − A positif berarti B lebih besar. Perbandingan ini deskriptif; perbedaan referensi, kondisi perekaman, waktu dan artefak perlu diperiksa. LLM tidak membuktikan emosi atau diagnosis.

## Jika ingin training

Buat template kosong dari Laboratorium model. Isi lokasi EDF, label yang terverifikasi, group subjek, awal dan durasi setiap segmen. Periksa kesiapan; training perlu setidaknya lima group serta kelas yang terwakili pada pembagian train/validation/test. Dua subjek yang tersedia belum memenuhi kebutuhan tersebut. beh.tsv berisi penilaian dan tidak otomatis menyediakan waktu segmen yang diperlukan. Demo sintetis hanya latihan alur, bukan bukti kemampuan emosi manusia.

## Pengujian

Jalankan UJI.bat. Smoke test tambahan memakai dua EDF asli lokal: `.venv\Scripts\python.exe -c "exec(open('tests/smoke_workflow.py',encoding='utf-8-sig').read())"`. Smoke test ini memerlukan unduhan EDF dan Tkinter.
