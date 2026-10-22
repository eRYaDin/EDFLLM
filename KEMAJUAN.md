# Kemajuan revisi — 23 Oktober 2026

Fokus baru: prediksi emosi tiap window EEG. Model perintah kecil buatan sendiri menentukan tugas; tujuh jaringan membaca fitur sinyal. Perbandingan sinyal menjadi fitur pendukung. Ollama/Gemma dipindahkan ke arsip dan tidak diimpor aplikasi baru.

## Sudah dibuat dan dijalankan

- Model neural intent dari bobot acak, corpus 154 kalimat (112/21/21 train/val/test), parser parameter, dispatcher.
- Tujuh modul jaringan terpisah, 14 eksperimen kedalaman, checkpoint tersimpan.
- Pemetaan 84 trial stimulus dari dua EDF asli, 64 kanal dinamis, fitur tiap window.
- Timeline prediksi, belum yakin, pencarian emosi, perbandingan emosi/sinyal, informasi EDF, ekspor.
- Interface Data → Perintah & hasil → Model & pelatihan; perubahan input membatalkan hasil lama.
- EDF dipindahkan ke data/raw; ukuran tetap 1.624.159.020 dan 1.549.761.444 byte. Empat pendamping ikut dipindahkan. Tidak diunduh ulang.
- Versi lama di archive/legacy-comparison-20261023 dan commit 2ea5999.

## Hasil awal

Intent uji 20/21 benar sebelum gate ketidakpastian. Dari 21 sampel uji, 16 diterima gate dan semuanya sesuai intent; lima ditolak (termasuk tiga permintaan di luar domain). Corpus kecil, bukan bahasa bebas. Model EEG terpilih dari validasi: tanh_baseline; akurasi subjek uji 16,2%, macro-F1 0,089. Sistem dapat dijalankan, tetapi pengenal emosi **belum andal**. Target kategori stimulus bukan kepastian perasaan aktual.

## Verifikasi

Lima kelompok unit test lulus: gradien tujuh jaringan, pemetaan/split trial, fitur dan kanal dinamis, routing/parameter, serta prediksi yang tidak dipengaruhi label anotasi. Smoke test EDF asli lulus untuk klasifikasi, pencarian, perbandingan emosi, perbandingan sinyal, metadata, render GUI, dan penghapusan hasil saat input berubah. Checksum SHA256 dua EDF setelah pemindahan disimpan lokal di output/edf-migration-check.json.

## Belum selesai

Peningkatan akurasi, validasi banyak subjek, kalibrasi skor, deteksi artefak, adapter dataset lain, target laporan subjektif. Evolution Algorithm masih ditunda. Latihan lengkap Bab 1–2 belum dikerjakan dalam revisi ini.

## Paket repository

Bobot dan laporan latihan tetap lokal di models/. Repo berisi kode latihan, corpus perintah, dokumentasi dan empat pendamping JSON/TSV. Checkpoint hasil latihan EEG tidak dipublikasikan. Pada salinan baru jalankan INSTALL.bat dan LATIH_EMOSI.bat setelah menyediakan EDF.
