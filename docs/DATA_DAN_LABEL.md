# Asal label dan pemisahan data

Profil khusus EmoEEG-MC sub-02/sub-03 berdasarkan README dataset yang diberikan pengguna dan sumber publik: [OpenNeuro ds005540](https://github.com/OpenNeuroDatasets/ds005540), [penulis dataset](https://github.com/ncclab-sustech/EmoEEG-MC).

Kedua subjek: TypeID 33 video, 32 imagery, 31 penilaian, 30 fade. EDF lokal masing-masing memuat tepat 21 video dan 21 imagery. Urutan stimulus berbeda per subjek; pemetaan memakai urutan ke-n **dalam context**, sesuai README. Tidak memakai urutan kelompok emosi atau kolom video_name TSV. Adapter menolak hitungan trigger yang berbeda; subjek lain tidak otomatis memakai profil ini.

## Pemotongan proyek

Batas kanan = trigger fade/rating/stimulus berikutnya. Awal fitur = onset + 1 detik. Durasi = minimum 28 detik dan batas kanan − 1 detik − awal. Durasi <4 detik ditolak. Window 4 detik tidak tumpang tindih; sisa pendek dibuang. Filter mendapat konteks sampai 2 detik.

Ini kebijakan konservatif proyek, **bukan reproduksi persis** pipeline penulis. Tidak ada penyambungan sampel melewati fade. Tabel hasil adapter ada di `output/dataset/trials.csv` setelah ekstraksi.

Target adalah kategori stimulus. Senang/sedih bukan pengamatan langsung perasaan tiap window. Empat pendamping tidak dipakai menebak waktu atau mengganti sinyal.

## Evaluasi

Train: sub-02 stimulus akhiran 4/5. Validasi: sub-02 stimulus 8. Test: sub-03. Video/imagery dari stimulus sama selalu satu split. Normalisasi hanya fit train; jaringan dipilih dari macro-F1 validasi. Nama file/subjek, context, label, nomor trial, onset bukan fitur prediksi.

Ini demonstrasi dua EDF, bukan validasi populasi. Prediksi subjek 02 diberi keterangan subjek latihan. Rekaman penuh termasuk istirahat/penilaian belum terwakili dalam target latihan.

Pengembangan: tambah subjek dengan profil label terverifikasi; evaluasi bergilir antar-subjek; preprocessing artefak; fitur spasial/temporal; target laporan subjektif; kalibrasi skor; corpus perintah dari pengguna baru. Jangan memilih model dengan test yang sama berulang kali.
