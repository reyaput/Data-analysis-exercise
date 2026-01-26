# 🏦 Bank Customer Churn Prediction System

## 📋 Table of Contents
1.  [Project Overview](#-project-overview)
2.  [Business Understanding](#-business-understanding)
3.  [Installation & Setup](#-installation--setup)
4.  [Data Understanding](#-data-understanding)
5.  [Key Insights (EDA)](#-key-insights-eda)
6.  [Machine Learning Model](#-machine-learning-model)
7.  [How to Run](#-how-to-run)
8.  [Business Recommendations](#-business-recommendations)

---

## 📋 Project Overview
Proyek ini bertujuan untuk membangun portofolio Data Analys serta membangun sistem *Machine Learning End-to-End* yang dapat memprediksi nasabah bank yang berisiko meninggalkan layanan (Churn). Dengan memanfaatkan data historis nasabah, sistem ini memberikan prediksi dini sehingga tim bisnis dapat melakukan intervensi (retensi) sebelum nasabah benar-benar pergi.

---

## 💼 Business Understanding

### Problem Statement
Bank menghadapi tingkat *Customer Churn* yang signifikan (~20%). Diketahui bahwa **biaya untuk mengakuisisi nasabah baru (Customer Acquisition Cost) adalah 5-25x lebih mahal** dibandingkan mempertahankan nasabah yang sudah ada. Kehilangan nasabah juga berarti hilangnya dana pihak ketiga (likuiditas) dan potensi pendapatan jangka panjang.

### Objectives
1.  **Identifikasi Faktor Risiko:** Menemukan pola perilaku atau demografi yang menyebabkan nasabah pergi.
2.  **Prediksi Dini:** Membangun model yang dapat mengklasifikasikan nasabah baru sebagai "Berisiko" atau "Aman".
3.  **Dashboard Operasional:** Menyediakan antarmuka (UI) sederhana bagi tim Customer Service untuk mengecek risiko nasabah secara *real-time*.

---

## 🛠️ Installation & Setup

Proyek ini dibangun menggunakan Python. Disarankan menggunakan **Conda** untuk manajemen environment agar dependensi terisolasi dengan rapi.

### 1. Prerequisite
Pastikan Anda telah menginstal [Anaconda](https://www.anaconda.com/) atau [Miniconda](https://docs.conda.io/en/latest/miniconda.html).

### 2. Setup Environment
Buka terminal (Anaconda Prompt / CMD) dan jalankan perintah berikut:

```bash
# 1. Buat environment baru bernama 'bank_churn' dengan Python 3.11
conda create -n bank_churn python=3.11 -y

# 2. Aktifkan environment
conda activate bank_churn

# 3. Masuk ke direktori project (Sesuaikan path folder Anda)
cd "path/to/your/project/folder"

```

### 3. Install Dependencies

Install library yang dibutuhkan menggunakan `pip`:

```bash
pip install -r requirements.txt

```

*(Pastikan file `requirements.txt` mencakup: pandas, numpy, scikit-learn, streamlit, matplotlib, seaborn, joblib)*

---

## 📊 Data Understanding

Dataset yang digunakan mencakup profil demografis, kepemilikan aset, dan aktivitas perbankan nasabah.

| Fitur | Deskripsi | Tipe Data |
| --- | --- | --- |
| **CreditScore** | Skor kredit nasabah. | Numerik |
| **Geography** | Negara domisili (France, Germany, Spain). | Kategorikal |
| **Gender** | Jenis kelamin. | Kategorikal |
| **Age** | Usia nasabah. | Numerik |
| **Tenure** | Lama menjadi nasabah (tahun). | Numerik |
| **Balance** | Saldo rekening saat ini. | Numerik |
| **NumOfProducts** | Jumlah produk bank yang digunakan. | Numerik |
| **HasCrCard** | Status kepemilikan kartu kredit (1=Ya, 0=Tidak). | Kategorikal |
| **IsActiveMember** | Status keaktifan nasabah (1=Aktif, 0=Pasif). | Kategorikal |
| **EstimatedSalary** | Estimasi gaji tahunan. | Numerik |
| **Complain** | Apakah nasabah pernah mengajukan keluhan? (1=Ya). | Kategorikal |
| **Satisfaction Score** | Skor kepuasan nasabah (1-5). | Numerik |
| **Point Earned** | Poin loyalitas yang dikumpulkan. | Numerik |
| **Exited (Target)** | Status churn (1=Ya/Churn, 0=Tidak/Stay). | Target |
| **AgeGroup** | Turunan Age | Kategorikal |
| **Tenure_Category** | Turunan Tenure | Kategorikal |
| **Balance_Category** | Turunan Balance | Kategorikal |
| **CreditScore_Category** | Turunan CreditScore | Kategorikal |

---

# 📊 Customer Churn Analysis Insights

## 1. Overview & Key Metrics
Secara keseluruhan, dataset menunjukkan adanya masalah retensi pelanggan yang cukup signifikan.
- **Total Churn:** Sebanyak kurang lebih **2.000 nasabah** (dari total 10.000) telah berhenti berlangganan.
- **Churn Rate:** Tingkat churn berada di angka **20.4%**. Angka ini tergolong tinggi untuk industri perbankan dan mengindikasikan 1 dari 5 nasabah memilih untuk meninggalkan bank.

## 2. Customer Profile at Risk (Siapa yang Churn?)
### a. Faktor Demografi & Geografi
* **Age Group (Senior):** Terdapat tren kenaikan probabilitas churn seiring bertambahnya usia. Nasabah kategori **Senior (>50 tahun)** jauh lebih rentan meninggalkan bank dibandingkan nasabah muda atau dewasa.
* **Geography (Germany):** Nasabah yang berdomisili di **Jerman** memiliki tingkat churn rate paling tinggi dibandingkan Prancis dan Spanyol. Hal ini mengindikasikan adanya masalah lokal di wilayah tersebut (bisa berupa kompetitor kuat atau layanan cabang yang kurang optimal).
* **Gender (Female):** Secara proporsi, nasabah **Perempuan** memiliki kecenderungan churn yang sedikit lebih tinggi dibandingkan laki-laki.

### b. Faktor Perilaku & Produk (Behavioral)
* **Product Holdings (3-4 Produk):** Terjadi anomali di mana nasabah yang memiliki **3 hingga 4 produk** justru memiliki tingkat churn yang sangat tinggi (hampir 100% pada 4 produk). Ini kontradiktif dengan asumsi umum bahwa "semakin banyak produk semakin setia". Kemungkinan terjadi *over-selling* atau nasabah merasa terbebani dengan kompleksitas produk.
* **Complain Status (Critical Factor):** Ini adalah indikator terkuat. Hampir **100% nasabah yang pernah mengajukan komplain (Complain = 1)** berakhir dengan status Churn.

## 3. Business Recommendations
Berdasarkan insight di atas, rekomendasi tindakan strategis adalah:
1.  **Immediate Complaint Handling:** Membangun sistem *priority response* untuk setiap komplain yang masuk, karena komplain adalah sinyal pasti nasabah akan pergi.
2.  **Germany Market Investigation:** Melakukan audit khusus terhadap layanan atau kompetitor di Jerman untuk memahami mengapa churn rate di sana sangat tinggi.
3.  **Review Product Bundling:** Mengevaluasi kembali strategi *bundling* produk. Jangan memaksa nasabah mengambil 3-4 produk jika itu justru membebani dan membuat mereka tidak nyaman.
4.  **Senior Citizen Program:** Merancang antarmuka aplikasi atau layanan yang lebih ramah bagi lansia (Senior Friendly) untuk menekan angka churn di demografi usia lanjut.

---

## 🤖 Machine Learning Model

### Algoritma: Random Forest Classifier

Dipilih sebagai model terbaik setelah perbandingan dengan Logistic Regression dan Decision Tree.

**Alasan Pemilihan:**

* **Non-Linearity:** Mampu menangkap pola kompleks (misal: Nasabah usia tua DI JERMAN lebih rentan).
* **Robustness:** Tahan terhadap outlier data keuangan.
* **Feature Importance:** Dapat menjelaskan faktor mana yang paling berpengaruh.

**Pipeline Training:**

1. **Preprocessing:**
* *Numerical:* Standard Scaler (Menyamakan skala angka).
* *Categorical:* One-Hot Encoding (Mengubah teks menjadi angka biner).


2. **Modeling:**
* Algoritma: Random Forest Classifier
* Estimators: 100 Trees
* Random State: 42 (untuk reproduktifitas)
* n_jobs : -1 

**Model Performance:**

* **Accuracy:** ~99.8% (Sangat tinggi, didorong dominasi fitur 'Complain').
* **Precision & Recall:** >99%.

---

## 🚀 How to Run

Proyek ini memiliki dua skrip utama yang dapat dijalankan secara independen.

### A. Melatih Model (Training)

Jalankan skrip ini jika Anda ingin melatih ulang model dengan data baru atau mereset model.

```bash
python train_model.py

```

*Output: Script akan memproses data, melatih model, menampilkan metrik evaluasi, dan menyimpan model ke file `churn_prediction_model.pkl`.*

### B. Menjalankan Dashboard (Deployment)

Jalankan perintah ini untuk membuka aplikasi prediksi berbasis web.

```bash
streamlit run app.py

```

*Output: Browser akan otomatis terbuka di `http://localhost:8501`. Anda bisa memasukkan data nasabah secara manual dan mendapatkan prediksi risiko churn.*

---

*Project created for Data Analysis Portfolio.*

```
