


# Iris Species – Traditional Machine Learning Project 

## 📌 Deskripsi Proyek
Proyek ini merupakan implementasi **end-to-end Data Science menggunakan Traditional Machine Learning** dengan dataset **Iris Species**.

Tujuan utama proyek:
- Menyediakan **template clean & modular** untuk proyek ML klasik
- Memisahkan dengan jelas **data, pipeline, model, experiment, dan report**
- Menjadi fondasi yang mudah dikembangkan ke skala production

Model baseline yang digunakan adalah **Logistic Regression**.

---

## 🧱 Arsitektur Proyek

Proyek ini mengikuti prinsip **Clean Architecture for Data Science**:

- `src/` → Core engine (data, model, pipeline)
- `scripts/` → CLI entrypoint (PowerShell)
- `experiments/` → Konfigurasi & catatan eksperimen
- `models_artifacts/` → Model & checkpoint
- `reports/` → Hasil evaluasi
- `data/` → Dataset (raw & processed)

Tidak ada dependency silang yang melanggar layer.

---

## 📂 Struktur Folder

```

project/
├── data/
│   ├── raw/iris.csv
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
├── reports/
├── scripts/
├── LICENSE
├── requirements.txt
└── README.md

````

---

## 📊 Dataset

- **Nama**: Iris Species Dataset
- **Format**: CSV
- **Lokasi**: `data/raw/iris.csv`
- **Target**: `species`
- **Fitur**:
  - sepal_length
  - sepal_width
  - petal_length
  - petal_width

Dataset **tidak boleh diubah** dan selalu dibaca dari folder `raw/`.

---

## ⚙️ Setup Environment

### 1️⃣ Buat Virtual Environment (opsional)
```powershell
python -m venv .venv
.\.venv\Scripts\activate
````

### 2️⃣ Install Dependency

```powershell
pip install -r requirements.txt
```

Dependency utama:

* pandas
* numpy
* scikit-learn
* joblib
* pyyaml

---

## ▶️ Cara Menjalankan Pipeline

### Training + Evaluation

```powershell
.\scripts\train.ps1
```

Pipeline ini akan:

1. Load data dari `data/raw/iris.csv`
2. Split data (train/val/test) secara stratified
3. Simpan hasil split ke `data/processed/`
4. Train Logistic Regression
5. Simpan model ke `models_artifacts/final/model.joblib`
6. Simpan metrik ke `reports/results.md`

---

### Evaluasi Saja

```powershell
.\scripts\evaluate.ps1
```