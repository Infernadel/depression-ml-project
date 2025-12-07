# 🧠 Depression Prediction Application - Panduan Lengkap

## 📋 Apa yang Akan Anda Lakukan?
1. Install library yang dibutuhkan
2. Simpan model dari hasil training
3. Jalankan aplikasi Streamlit
4. Mulai prediksi!

---

## 🚀 TAHAP 1: Install Dependencies

Buka terminal/command prompt, lalu jalankan:

```bash
pip install streamlit pandas numpy scikit-learn
```

**Catatan:** 
- `pickle` sudah built-in di Python, tidak perlu install
- Jangan install `pickle5` (akan error di Windows)

**Jika menggunakan Anaconda:**
```bash
conda install streamlit pandas numpy scikit-learn
```

---

## 💾 TAHAP 2: Simpan Model dari Training

### Langkah 2.1: Buka Script Training Anda
Buka file `train_model.py` (atau nama file training Anda)

### Langkah 2.2: Scroll ke Paling Bawah
Cari bagian akhir script (setelah section "10. PREDIKSI & EVALUASI")

### Langkah 2.3: Copy-Paste Kode Berikut

**COPY** semua kode dari artifact **"Kode untuk Menyimpan Model"** dan **PASTE** di akhir script training Anda.

Kode tersebut akan:
- ✅ Menggabungkan preprocessor + model jadi satu pipeline
- ✅ Menyimpan ke file `depression_model.pkl`
- ✅ Melakukan test prediksi
- ✅ Menampilkan konfirmasi berhasil

### Langkah 2.4: Jalankan Script Training

```bash
python train_model.py
```

Jika berhasil, Anda akan melihat output seperti ini:

```
💾 MENYIMPAN MODEL UNTUK APLIKASI STREAMLIT
============================================================
✓ Fitur numerik: 5 fitur
✓ Fitur kategorikal: 7 fitur
✓ Pipeline dibuat: Preprocessor + Logistic Regression
⏳ Fitting pipeline dengan seluruh data...
✓ Pipeline berhasil di-fit!
✓ Akurasi model dalam pipeline: 0.8523
✓ Model berhasil disimpan!
  📁 File: depression_model.pkl
  📊 Ukuran: 234.56 KB
✓ Test prediksi berhasil!
✅ MODEL SIAP DIGUNAKAN UNTUK APLIKASI STREAMLIT!
```

**Cek:** Pastikan file `depression_model.pkl` sudah ada di folder project Anda!

---

## 🎨 TAHAP 3: Jalankan Aplikasi Streamlit

### Langkah 3.1: Pastikan Struktur Folder Benar

```
your-project-folder/
│
├── app.py                          # Aplikasi Streamlit
├── train_model.py                  # Script training (sudah dijalankan)
├── depression_model.pkl            # Model yang sudah disimpan ✅
└── final_depression_dataset_1.csv  # Dataset asli
```

### Langkah 3.2: Jalankan Aplikasi

Buka terminal di folder project, lalu jalankan:

```bash
streamlit run app.py
```

### Langkah 3.3: Buka Browser

Aplikasi akan otomatis terbuka di browser dengan alamat:
```
http://localhost:8501
```

Jika tidak otomatis terbuka, copy URL tersebut dan paste di browser Anda.

---

## 🎯 TAHAP 4: Gunakan Aplikasi

### 1. Isi Formulir Input

Aplikasi memiliki beberapa section:

#### 👤 **Informasi Demografis**
- **Age**: Usia Anda (10-100 tahun)
- **Gender**: Male atau Female
- **Combined Profession**: Pilih profesi dari dropdown (36 pilihan)
- **Degree**: Pilih pendidikan dari dropdown (27 pilihan)

#### 💼 **Aktivitas & Tekanan**
- **Work/Study Hours**: Jam kerja/belajar per hari (0-24)
- **Financial Stress**: Skala 1-5
- **Pressure Score**: Skala 0-10
- **Overall Satisfaction**: Skala 1-5

#### 🌙 **Gaya Hidup**
- **Sleep Duration**: Durasi tidur per malam
- **Dietary Habits**: Healthy/Moderate/Unhealthy

#### 🏥 **Riwayat Kesehatan Mental**
- **Have You Ever Had Suicidal Thoughts?**: Yes/No
- **Family History of Mental Illness**: Yes/No

### 2. Klik "🔮 Predict Depression"

### 3. Lihat Hasil Prediksi

Aplikasi akan menampilkan:
- ✅ **Status**: Depressed atau Not Depressed
- 📊 **Probabilitas**: Persentase kemungkinan
- 📈 **Visualisasi**: Progress bar
- 💡 **Rekomendasi**: Saran berdasarkan hasil

---

## 🔧 Troubleshooting (Jika Ada Masalah)

### ❌ Error: "Model file tidak ditemukan"
**Penyebab:** File `depression_model.pkl` tidak ada

**Solusi:**
1. Pastikan Anda sudah jalankan TAHAP 2
2. Cek apakah file `depression_model.pkl` ada di folder yang sama dengan `app.py`
3. Jika belum ada, jalankan ulang script training dengan kode save model

---

### ❌ Error: "ModuleNotFoundError: No module named 'streamlit'"
**Penyebab:** Streamlit belum terinstall

**Solusi:**
```bash
pip install streamlit
```

---

### ❌ Error di VS Code: "Import could not be resolved"
**Penyebab:** Warning dari Pylance (linter), bukan error fatal

**Solusi:** 
- **Abaikan saja!** Kode tetap bisa dijalankan
- Atau set Python interpreter yang benar:
  1. Tekan `Ctrl + Shift + P`
  2. Ketik "Python: Select Interpreter"
  3. Pilih interpreter yang sudah ada library-nya

---

### ❌ Aplikasi lambat atau hang
**Solusi:**
- Pastikan file model tidak terlalu besar (< 10 MB)
- Restart aplikasi: `Ctrl + C` di terminal, lalu `streamlit run app.py` lagi
- Clear cache: Klik menu ☰ di kanan atas → "Clear cache"

---

### ❌ Prediksi selalu sama
**Penyebab:** Model belum di-fit dengan benar

**Solusi:**
1. Cek apakah akurasi model sudah bagus di training (> 70%)
2. Pastikan data training cukup besar dan balanced
3. Coba training ulang dengan data yang lebih bersih

---

## 📊 Fitur Aplikasi

✅ **36 Profesi** dari dataset asli  
✅ **27 Degree** dari dataset asli  
✅ **Input Validation** - tolak input tidak valid  
✅ **Autocomplete Dropdown** - semua kategori  
✅ **Bilingual** - field Inggris, penjelasan Indonesia  
✅ **Real-time Prediction** - hasil langsung  
✅ **Probability Visualization** - progress bar & metrics  
✅ **Smart Recommendations** - saran berbeda per hasil  
✅ **Mental Health Hotline** - kontak darurat Indonesia  

---

## 📚 Informasi Tambahan

### Kategori yang Tersedia

**Combined Profession (36 pilihan):**
- Student, Teacher, Doctor, Lawyer, Accountant
- Software Engineer, Data Scientist, UX/UI Designer
- Mechanical Engineer, Civil Engineer, Architect
- Manager, HR Manager, Marketing Manager
- Business Analyst, Financial Analyst, Consultant
- Dan 20+ profesi lainnya...

**Degree (27 pilihan):**
- Bachelor: B.Tech, BE, BBA, BCA, BA, B.Com, BSc, dll
- Master: M.Tech, MBA, MCA, MA, M.Com, MSc, dll
- Professional: LLB, LLM, MBBS, MD, PhD
- Class 12

---

## ⚠️ Disclaimer

**Aplikasi ini BUKAN pengganti diagnosis medis profesional!**

Untuk diagnosis dan penanganan yang tepat, konsultasikan dengan:
- Psikolog
- Psikiater  
- Konselor profesional
- Dokter spesialis kesehatan mental

### Hotline Kesehatan Mental Indonesia:
- **Sehatmental.id (Kemenkes)**: 119 ext. 8
- **Into The Light**: 021-7884-5855
- **LSM Jangan Bunuh Diri**: 021-9696-9293

---

## 🎉 Selamat Mencoba!

Jika semua tahap sudah diikuti dengan benar, aplikasi Anda sudah siap digunakan!

**Tips:**
- Coba berbagai kombinasi input untuk melihat bagaimana model memprediksi
- Perhatikan probabilitas, bukan hanya label prediksi
- Gunakan untuk edukasi, bukan untuk self-diagnosis

---

## 📞 Butuh Bantuan?

Jika masih ada masalah:
1. ✅ Pastikan semua library terinstall
2. ✅ Pastikan file `depression_model.pkl` ada
3. ✅ Pastikan app.py dan model.pkl dalam folder yang sama
4. ✅ Cek error message di terminal untuk detail

**Selamat menggunakan aplikasi! 💚**