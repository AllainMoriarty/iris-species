# Iris Species – Traditional Machine Learning Project

## Deskripsi Proyek

Proyek ini merupakan implementasi **end-to-end Data Science menggunakan Traditional Machine Learning** dengan dataset **Iris Species**.

Tujuan utama proyek:

* Menyediakan template **clean, modular, dan reproducible** untuk proyek Machine Learning klasik
* Memisahkan secara tegas **data, pipeline, model, experiment, dan report**
* Menjadi fondasi yang mudah dikembangkan ke skala production

Model baseline yang digunakan adalah **Logistic Regression**, dengan dukungan eksperimen tambahan seperti **KNN** dan **SVM** melalui konfigurasi.

Setiap model **disimpan secara terpisah berdasarkan nama model** untuk menjamin reproducibility dan menghindari overwrite antar eksperimen.

---

## Arsitektur Proyek

Proyek ini mengikuti prinsip **Clean Architecture for Data Science**:

* `src/` → Core engine (data processing, model, pipeline)
* `scripts/` → CLI entrypoint (PowerShell / Bash)
* `experiments/` → Konfigurasi dan catatan eksperimen
* `models_artifacts/` → Model hasil training dan checkpoint
* `reports/` → Hasil evaluasi model
* `data/` → Dataset (raw dan processed)

Setiap layer bersifat independen dan **tidak saling melanggar dependency**.

---

## Struktur Folder

```
project/
├── data/
│   ├── raw/
│   │   └── iris.csv
│   └── processed/
│
├── src/
│   ├── config/
│   ├── data/
│   ├── models/
│   ├── pipelines/
│   └── utils/
│
├── experiments/
├── models_artifacts/
│   └── final/
│       ├── logistic_regression.joblib
│       ├── knn.joblib
│       └── svm.joblib
├── reports/
├── scripts/
├── LICENSE
├── requirements.txt
└── README.md
```

---

## Dataset

* Nama: Iris Species Dataset
* Format: CSV
* Lokasi: `data/raw/iris.csv`
* Target column: `species`
* Features:

  * sepal_length
  * sepal_width
  * petal_length
  * petal_width

Dataset **bersifat read-only** dan **tidak boleh dimodifikasi**. Semua proses dilakukan pada data hasil split di folder `processed/`.

---

## Setup Environment

### 1. Virtual Environment (opsional)

```powershell
python -m venv .venv
.\.venv\Scripts\activate
```

### 2. Install Dependency

```powershell
pip install -r requirements.txt
```

Library utama yang digunakan:

* pandas
* numpy
* scikit-learn
* joblib
* pyyaml

---

## Menjalankan Pipeline

### Training dan Evaluation

```powershell
.\scripts\train.ps1
```

Pipeline ini akan menjalankan tahapan berikut:

1. Load dataset dari `data/raw/iris.csv`
2. Split data train / validation / test secara stratified
3. Simpan hasil split ke `data/processed/`
4. Training model sesuai konfigurasi eksperimen
5. Simpan model ke `models_artifacts/final/<model_name>.joblib`
6. Simpan metrik evaluasi ke `reports/results.md`

### Menjalankan Eksperimen Model

Logistic Regression:

```powershell
.\scripts\train.ps1 experiments\exp_001_logreg\config.yaml
```

KNN:

```powershell
.\scripts\train.ps1 experiments\exp_002_knn\config.yaml
```

SVM:

```powershell
.\scripts\train.ps1 experiments\exp_003_svm\config.yaml
```

---

### Evaluasi Saja
Logistic Regression:

```powershell
.\scripts\evaluate.ps1 experiments\exp_001_logreg\config.yaml
```

KNN:

```powershell
.\scripts\evaluate.ps1 experiments\exp_002_knn\config.yaml
```

SVM:

```powershell
.\scripts\evaluate.ps1 experiments\exp_003_svm\config.yaml
```

Evaluasi **selalu menggunakan model sesuai experiment config**.
