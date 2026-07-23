# Energeek Box Detector 

Sebuah *production-grade web application* yang menggunakan **Classical Computer Vision** untuk secara otomatis mendeteksi, mengklasifikasi, dan menghitung jumlah kotak biru yang berada di dalam maupun di luar dari kotak referensi utama (kotak putih).

---

## Apa Itu Energeek Box Detector?

Aplikasi ini dibuat untuk menyelesaikan masalah inspeksi industri secara otomatis. Diberikan sebuah gambar yang berisi satu kotak utama berwarna putih dengan garis tepi hitam, serta kumpulan kotak-kotak kecil berwarna biru (baik itu kotak solid murni maupun kotak dengan tekstur bergaris/hatch).

Aplikasi akan:
1. Menemukan koordinat presisi dan kemiringan (*rotation*) dari kotak utama (putih).
2. Mendeteksi seluruh kotak biru di dalam gambar.
3. Mengkategorikan posisi kotak biru tersebut ke dalam 3 jenis:
   - **Inside** (Hijau): Murni berada di dalam batas area kotak putih.
   - **Outside** (Merah): Murni berada di luar batas area kotak putih.
   - **Intersecting / Kepotong** (Abu-abu): Objek menabrak/memotong garis batas kotak putih (sengaja diabaikan/tidak dihitung).
4. Menghasilkan *overlay image* dengan *bounding box*, titik tengah (*centroid*), label klasifikasi, dan total perhitungan.

---

## Teknologi yang Digunakan

Aplikasi ini sengaja dibangun dengan konsep *Full-Stack Minimal-Dependency* untuk menjaga performa tetap super cepat (*ultra-low latency*).

| Komponen | Teknologi | Alasan Pemilihan |
| :--- | :--- | :--- |
| **Backend API** | **FastAPI** (Python) | Dipilih karena sangat cepat (*async-native*), memiliki *auto-documentation* bawaan (Swagger UI), validasi data otomatis menggunakan Pydantic, dan sangat cocok untuk microservices berbasis *Machine Learning* / *Computer Vision*. |
| **Computer Vision** | **OpenCV** (`cv2`) | Menggunakan pendekatan **Classical CV** (bukan *Deep Learning*) karena karakteristik gambar memiliki kontras warna yang solid dan latar yang stabil. OpenCV mampu memproses logika *thresholding* dan pencarian kontur geometri dalam hitungan milidetik (~50ms) tanpa membutuhkan GPU yang mahal. |
| **Frontend UI** | **Vanilla HTML, CSS, JS** | Tidak menggunakan framework berat seperti React/Vue. Didesain dalam 1 file `index.html` saja dengan *Vanilla JS* untuk mempermudah operasional dan meminimalisir *setup*. Tampilan menggunakan estetika modern (Glassmorphism, transisi halus) secara murni dengan CSS Variables. |

---

## Cara Menggunakan Aplikasi (Usage Guide)

### 1. Clone Repository
Pertama, salin kode program ini ke komputer Anda dan masuk ke dalam direktorinya:
```bash
git clone https://github.com/your-repo/energeek-box-detector.git
cd energeek-box-detector
```

### 2. Install Dependencies
Pastikan Anda memiliki Python 3.10 atau versi di atasnya terinstall. Instal pustaka yang dibutuhkan menggunakan `pip`:
```bash
pip install -r requirements.txt
```

### 3. Jalankan Server (Syntax Run)
Gunakan Uvicorn untuk menjalankan FastAPI backend. Syntax di bawah ini akan menjalankan server di port 8000 dan mengaktifkan mode *auto-reload* untuk *development*.
```bash
uvicorn app.main:app --reload --port 8000
```

### 4. Buka Frontend Web UI
Setelah server menyala (ditandai dengan pesan `Application startup complete` di terminal), buka browser Anda dan akses:
> **http://localhost:8000/**

### 5. Cara Penggunaan Web UI
1. Anda akan melihat halaman antarmuka (*dashboard*).
2. Di kotak sebelah kiri, Anda bisa melakukan **Drag & Drop** file gambar atau klik untuk memilih gambar (file harus berupa format `.png`, `.jpg`, atau `.jpeg`).
3. Gambar akan langsung terproses. Di layar sebelah kanan, Anda akan mendapatkan kalkulasi hitungan (Inside, Outside, Total) dan gambar visualisasi hasilnya lengkap dengan warna anotasi (Hijau, Merah, Abu-abu).

---

## Sederhana, Bagaimana Pipeline (Cara Kerja) Sistem Ini?

Aplikasi ini menggunakan sistem **Hybrid Pipeline** yang menggabungkan dua algoritma cerdas untuk dapat mengenali objek bergaris maupun objek solid murni.

Berikut adalah diagram alur sederhana (*pipeline*) dari proses yang terjadi di belakang layar:

```text
       [ Input Image ]
              |
              v
    ( Image Preprocessing )
  Resize max 1024px & Denoising
              |
      +-------+-------+
      |       |       |
      v       v       v
 White Box    |  Colored Object
 Detection    |   Segmentation
 (Threshold)  |    (HSV Mask)
      |       |       |
      |       v       |
      | Black Outline |
      |  Extraction   |
      |  (Threshold)  |
      +-------+-------+
              |
              v
     Object Validation 
 (Size & Position Filtering)
              |
              v
   Merge Detection Result 
   (Overlap elimination)
              |
              v
   Position Classification
 (Inside / Outside / Ignore)
              |
              v
      Visual Annotation 
    (Draw bounding boxes)
              |
              v
    [ Return to Frontend ]
```

### Langkah-langkah Detail (*Step-by-Step*):
1. **Preprocess**: Gambar di-*resize* ke ukuran maksimal tertentu agar beban komputasi stabil, kemudian diberikan efek *blur* halus untuk menghilangkan *noise*.
2. **Segmentasi Referensi (White Box)**: OpenCV mencari batas terluar warna hitam (`Threshold < 60`) dan menggunakan hierarki berlapis (`RETR_CCOMP`) untuk mengambil kontur poligon persegi panjang terbesar sebagai landasan kotak referensi.
3. **Hybrid Blue Box Segmentation**: 
   - *Algoritma 1*: Mencari objek biru dengan mengidentifikasi pola *outline* hitam di sekelilingnya (sangat akurat untuk kotak bertipe arsiran/garis).
   - *Algoritma 2*: Mencari objek berdasarkan *masking* HSV untuk warna biru murni (sangat akurat untuk kotak polos tanpa outline hitam).
   - Keduanya digabung sambil memastikan tidak ada objek yang dihitung dobel menggunakan metode *bitwise_and overlap checking*.
4. **Klasifikasi Matematika**: Fungsi `cv2.pointPolygonTest` di OpenCV digunakan untuk mengecek apakah titik *centroid* (titik tengah pasti) dari sebuah objek biru itu ada di dalam kemiringan kotak putih atau tidak. Jika kontur objek mengenai garis luar kotak putih, akan ditandai *Intersecting* dan dikeluarkan dari hitungan total.
5. **Base64 Rendering**: Daripada menyimpan gambar di *hard disk* server setiap ada unggahan, sistem langsung menggambar anotasi di *memory RAM* dan mengubahnya menjadi *string Base64* untuk dikirim ke browser.

