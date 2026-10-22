# Kemajuan EEG EDF Lab — 22 Oktober 2026

## Hasil yang sudah dibuat

- Folder aplikasi mandiri dengan environment .venv, launcher, installer, GUI dan CLI.
- Pembaca EDF/EDF+ kontinu, header/anotasi dan TSV pendamping.
- Perbandingan dua rekaman, kanal bersama, satuan uV, resampling, filter, statistik dan daya pita.
- Panel metadata, tabel, grafik tiap kanal, ringkasan serta ekspor JSON/CSV/PNG.
- Tujuh file model terpisah, registry pemanggil, base layer bersama, pipeline training dan pemuatan model.
- Perbandingan satu versus tiga hidden layer, split per group, normalisasi hanya dari train, early stopping, penyimpanan metrik/kurva, waktu pelatihan dan jumlah parameter.
- Penjelasan opsional dari model Ollama lokal atas hasil perhitungan.
- Demo sintetis, manifest contoh, model demo, dan panduan penggunaan.

## Bukti verifikasi

Delapan kelompok pengujian lulus: header/anotasi EDF, konversi satuan, input tidak valid, arah perbedaan PSD/ekspor, pita tidak tersedia, pemisahan group, numerical gradient tujuh model, serta training/save/load ketujuhnya.

GUI berhasil dibuat dan diuji dengan pemuatan metadata dua kanal, pengisian 18 baris perbandingan, pembangunan grafik, serta enam tab. Ini uji dasar program, bukan seluruh interaksi manual pengguna.

Demo membandingkan A (160 Hz, uV, alpha kuat) dan B (200 Hz, mV, beta kuat). Setelah preprocessing sama, arah perubahan alpha/beta sesuai sinyal pembuatnya. Gambar ekspor telah diperiksa secara visual.

Ketujuh model baseline dan deep berhasil dilatih 120 epoch maksimum pada demo, disimpan ke output/models/demo, lalu model ReLU dimuat ulang untuk prediksi A/B. Demo menghasilkan accuracy dan macro F1 1.0 pada holdout sintetis yang mudah. Angka ini hanya memeriksa alur program, bukan performa EEG manusia, dan tidak membuktikan satu aktivasi lebih unggul.

Folder output/demo_compare berisi laporan dan grafik. Folder demo berisi rekaman buatan, bukan salinan EDF milik pengguna. Tidak ada data EEG manusia yang dipakai untuk melatih model ini.

## Belum diverifikasi atau dikembangkan

1. Pembacaan dan evaluasi pada EDF asli pengguna, termasuk file nonstandar dan ukuran besar.
2. Model pengenal emosi EmoEEG-MC. Diperlukan pemetaan trial/label, preprocessing sesuai dataset, dan evaluasi antar-subjek/konteks.
3. Pemetaan otomatis trigger/events menjadi segmen, validasi missing trial, dan pemuatan fitur derivatives .npy.
4. Referensi elektroda, montase, ICA, interpolasi kanal dan evaluasi kualitas sinyal yang tervalidasi.
5. Prediksi LLM lokal belum diuji end-to-end. Modul hanya menjelaskan tabel atas permintaan; bukan router training/model mandiri.
6. Pembatalan worker/training, ketahanan model di luar distribusi, statistik inferensial serta evaluasi performa pengguna nyata.
7. Evolution Algorithm ditunda. Latihan lengkap Bab 1–2 bukan cakupan aplikasi versi ini.

## Prioritas berikutnya

Pilih satu pasangan EDF + events asli, verifikasi pembacaan dan keselarasan label, lalu susun manifest kecil dengan kelompok subjek yang benar. Gunakan model yang sama untuk kedua rekaman dan bandingkan hasil pada konteks eksperimen yang sesuai. Hindari menganggap setiap selisih power sebagai perubahan emosi.

## Perbaikan 23 Oktober 2026 — manifest kosong

Kolom manifest kosong sebelumnya diterjemahkan Path('') menjadi direktori kerja, lalu program mencoba membuka folder sebagai CSV dan memunculkan Permission denied di Windows. Kini input kosong, folder, file tidak ada, ekstensi salah, serta TSV perilaku tanpa skema manifest ditolak dengan pesan yang menjelaskan langkah berikutnya. GUI memvalidasi path sebelum membuat worker training. Sembilan kelompok tes lulus setelah perbaikan, termasuk regresi kasus ini.

### Kanal training terpisah

Training sebelumnya mengambil kanal yang dipilih pada perbandingan A/B. Jika manifest demo dipilih saat Fp1 dari data asli masih terpilih, pembacaan demo gagal karena hanya berisi C3/C4. Kini ada kolom kanal training sendiri dengan default auto berdasarkan irisan kanal tegangan seluruh EDF manifest. Kanal eksplisit diperiksa sebelum ekstraksi fitur. Sepuluh tes lulus; uji regresi GUI memastikan training memakai C3/C4 meski kanal perbandingan tetap Fp1.

### 23 Oktober 2026 — kanal mengikuti EDF pengguna
Pemilihan kanal pembanding dan training kini memakai fungsi bersama yang membaca header EDF, tanpa daftar elektroda tetap. Semua kanal tegangan bersama dipilih otomatis. Tombol Deteksi kanal dari manifest memperlihatkan kanal sumber training; contoh C3/C4 di kontrol utama dihapus.
Dua EDF asli sub-02/sub-03 berhasil dibaca: masing-masing 64 kanal, sampling 1200 Hz, durasi 10566/10082 detik. Perbandingan 10 detik seluruh kanal menghasilkan 576 baris metrik. Sebelas tes dan uji GUI pemilihan 64 kanal lulus. Tes tambahan memakai EDF dengan kanal Fz/Pz, tanpa kanal demo.
Training emosi pada EDF asli belum dilakukan; pemetaan segmen/label dan split subjek masih diperlukan. Demo tetap hanya bahan pengujian opsional.

## Revisi alur antarmuka — 23 Oktober 2026

- Alur utama empat langkah: EDF → segmen → hasil → Tanya AI. Training/prediksi dipisah dalam Laboratorium model.
- Persyaratan file dijelaskan per halaman; JSON/TSV opsional ditemukan otomatis, manifest hanya untuk training.
- Hasil default berupa penjelasan dan angka bersatuan per kanal; rincian mentah tersedia terpisah.
- Kanal mengikuti header EDF; perubahan input/pengolahan membatalkan hasil dan percakapan lama.
- LLM menerima hasil hitung kanal terpilih dan pertanyaan; koneksi Ollama dapat diperiksa. Belum menjadi router mandiri tujuh model.
- Validasi: 11 tes inti lulus; smoke GUI memakai EDF asli sub-02/sub-03 berhasil dengan 64 kanal dan 576 metrik. Respons model Ollama nyata belum diuji pada revisi ini.
- Git mengecualikan EDF, environment, model/output sementara dan arsip. Empat pendamping JSON/TSV disertakan.
