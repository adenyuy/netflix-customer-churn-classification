# 📺 Prediksi Churn Pelanggan Netflix dengan Machine Learning

Selamat datang di proyek Prediksi Churn Pelanggan Netflix! Proyek *end-to-end* ini bertujuan untuk menganalisis data pelanggan, membangun model *machine learning* untuk memprediksi pelanggan yang berpotensi berhenti berlangganan (*churn*), dan menyajikannya dalam sebuah aplikasi web interaktif.

## 📝 Deskripsi Proyek

Di industri layanan berlangganan, mempertahankan pelanggan yang ada jauh lebih hemat biaya daripada mengakuisisi pelanggan baru. Proyek ini mengatasi masalah tersebut dengan memanfaatkan data historis pelanggan untuk mengidentifikasi pola perilaku yang mengarah ke *churn*. Dengan model ini, tim bisnis dapat secara proaktif menargetkan pelanggan yang berisiko dengan penawaran retensi sebelum mereka memutuskan untuk pergi.

Proyek ini mencakup seluruh alur kerja ilmu data, mulai dari analisis data eksplorasi (EDA), *preprocessing*, pelatihan model, hingga *deployment* model sebagai aplikasi web menggunakan Streamlit.

## ✨ Fitur Utama

* **Analisis Data Mendalam:** Eksplorasi data di Jupyter Notebook untuk menemukan wawasan dan korelasi antar fitur.
* **Perbandingan Model:** Implementasi dan evaluasi dua model klasifikasi yang kuat: **Random Forest** (sebagai *baseline*) dan **XGBoost** (sebagai model lanjutan).
* **Evaluasi Komprehensif:** Kinerja model diukur menggunakan Laporan Klasifikasi, *Confusion Matrix*, dan kurva ROC/AUC untuk memastikan keandalan.
* **Aplikasi Web Interaktif:** Antarmuka pengguna yang dibangun dengan Streamlit memungkinkan prediksi *real-time* dengan memasukkan data pelanggan.

## 🛠️ Tech Stack & Library

* **Bahasa:** Python 3.11+
* **Analisis Data:** Pandas, NumPy
* **Visualisasi Data:** Matplotlib, Seaborn
* **Machine Learning:** Scikit-learn, XGBoost
* **Deployment Aplikasi:** Streamlit
* **Manajemen Model:** Joblib

## 📂 Struktur Proyek

```
/
|-- app.py                    # Skrip utama aplikasi Streamlit
|-- train_models.py           # Skrip untuk melatih ulang model dari awal
|-- analisis_churn_netflix.ipynb  # Notebook analisis dan eksplorasi data
|-- requirements.txt          # Daftar library yang dibutuhkan
|-- netflix_customer_churn.csv  # Dataset yang digunakan
|-- README.md                 # Dokumentasi proyek
|-- .gitignore                # File yang diabaikan oleh Git
```

## 📂 Dokumentasi
![Screenshot (883)](https://github.com/user-attachments/assets/144d52e1-3a82-4496-b293-81aac5a8e543)

