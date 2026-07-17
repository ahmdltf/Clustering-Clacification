# 🧠 Customer Segmentation & Classification using Machine Learning

<div align="center">

![Python](https://img.shields.io/badge/Python-3.12-blue?style=for-the-badge&logo=python)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange?style=for-the-badge&logo=scikitlearn)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?style=for-the-badge&logo=pandas)
![NumPy](https://img.shields.io/badge/NumPy-Numerical-013243?style=for-the-badge&logo=numpy)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-blue?style=for-the-badge)
![Seaborn](https://img.shields.io/badge/Seaborn-EDA-4C72B0?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

**An End-to-End Machine Learning Project for Customer Segmentation and Classification**

*From Exploratory Data Analysis → Data Preprocessing → Clustering → Cluster Interpretation → Classification Model Development*

</div>

---

# 📖 Table of Contents

- [🧠 Proyek Utama](#-customer-segmentation--classification-using-machine-learning)
- [🚀 Project Overview](#-project-overview)
- [🎯 Business Understanding](#-business-understanding)
- [🎯 Objectives](#-objectives)
- [📊 Dataset Information](#-dataset-information)
- [🔄 Machine Learning Workflow](#-machine-learning-workflow)
- [🔍 Exploratory Data Analysis](#-exploratory-data-analysis)
- [💼 Business Impact & Data-Driven Insights](#-business-impact--data-driven-insights)
- [📊 Machine Learning Pipeline](#-machine-learning-pipeline)
- [📦 Generated Artifacts](#-generated-artifacts)
- [🚀 How to Run](#-how-to-run)
- [📚 Libraries](#-libraries)
- [🎯 Learning Outcomes](#-learning-outcomes)
- [👨‍💻 Author](#-author)
- [⭐ Repository Goals](#-repository-goals)

---

# 🚀 Project Overview

This repository demonstrates the complete implementation of a supervised and unsupervised Machine Learning workflow for customer data analysis.

Instead of directly building a predictive model, this project first discovers hidden customer patterns through **unsupervised learning (K-Means Clustering)**. The generated cluster labels are then utilized as target classes to develop a **supervised classification model**, allowing future customer segmentation without performing clustering repeatedly.

The project follows an end-to-end data science workflow consisting of:

- Exploratory Data Analysis (EDA)
- Data Cleaning
- Feature Engineering
- Feature Encoding
- Feature Scaling
- Outlier Handling
- Feature Binning
- K-Means Clustering
- Cluster Evaluation
- PCA Visualization
- Cluster Interpretation
- Decision Tree Classification
- Alternative Classification Model
- Hyperparameter Tuning
- Model Serialization

The implementation follows industry-standard machine learning practices using the Scikit-Learn ecosystem and reproducible preprocessing pipelines.

---

# 🎯 Business Understanding

Customer datasets usually contain dozens of behavioral, demographic, and transactional variables.

Although the data is rich, organizations often face several challenges:

- Customers are treated as one homogeneous population.
- Marketing campaigns are distributed equally to all customers.
- Loyalty programs fail because they ignore customer behavior.
- Resource allocation becomes inefficient.
- Customer lifetime value is difficult to maximize.

Without customer segmentation, businesses cannot understand:

- Which customers are highly valuable?
- Which customers are likely to churn?
- Which customers respond to promotions?
- Which customers require different marketing strategies?

Machine Learning provides an effective solution by discovering natural customer groups based on their characteristics.

Rather than manually defining customer categories, clustering algorithms learn hidden structures directly from data.

These discovered customer segments can later become labels for supervised classification models, enabling automatic prediction of customer segments for future data.

---

# 🎯 Objectives

This project aims to:

- Perform comprehensive exploratory data analysis.
- Clean and preprocess raw customer data.
- Handle missing values and duplicated records.
- Encode categorical variables.
- Detect and remove outliers.
- Normalize numerical variables.
- Apply feature engineering through binning.
- Build customer segments using K-Means Clustering.
- Determine the optimal number of clusters using the Elbow Method.
- Evaluate clustering quality using Silhouette Score.
- Visualize clusters using Principal Component Analysis (PCA).
- Interpret cluster characteristics based on original feature values.
- Generate labeled datasets for supervised learning.
- Develop customer classification models.
- Compare multiple classification algorithms.
- Improve model performance using Hyperparameter Tuning.
- Save trained models for future deployment.

---

# 📊 Dataset Information

The project uses a structured tabular dataset containing customer-related information.

The dataset consists of numerical and categorical variables describing customer profiles and behaviors.

Typical feature categories include:

### Numerical Features

- Age
- Income
- Spending Score
- Purchase Frequency
- Transaction Amount
- Product Quantity

### Categorical Features

- Gender
- Occupation
- Region
- Customer Category
- Payment Method
- Membership Status

During preprocessing, identifier columns that do not contribute to predictive performance are removed, including:

- Customer ID
- Transaction ID
- Device ID
- IP Address
- Date columns

These variables serve only as identifiers and do not contain meaningful information for clustering or classification.

---

# 🔄 Machine Learning Workflow

The complete workflow implemented in this project is illustrated below.

```text
                Raw Dataset
                     │
                     ▼
         Exploratory Data Analysis
                     │
                     ▼
             Data Cleaning
      ├── Missing Values
      ├── Duplicate Removal
      ├── Remove ID Columns
                     │
                     ▼
         Feature Engineering
      ├── Label Encoding
      ├── Outlier Removal
      ├── Standard Scaling
      ├── Feature Binning
                     │
                     ▼
          Preprocessed Dataset
                     │
        ┌────────────┴────────────┐
        ▼                         ▼
 K-Means Clustering        PCA Visualization
        │
        ▼
  Cluster Evaluation
  ├── Elbow Method
  ├── Silhouette Score
        │
        ▼
 Cluster Interpretation
        │
        ▼
 Generate Target Labels
        │
        ▼
 Supervised Classification
        │
        ▼
Decision Tree
Random Forest
Hyperparameter Tuning
        │
        ▼
Saved Machine Learning Models
```

---

# 🔍 Exploratory Data Analysis

Exploratory Data Analysis (EDA) is performed to understand the structure, quality, and statistical characteristics of the dataset before model development.

Several analyses are conducted during this phase.

### Dataset Inspection

The notebook begins by displaying the first few observations using:

```python
df.head()
```

This step verifies that the dataset has been successfully loaded and provides an initial understanding of the available features.

---

### Dataset Information

The dataset structure is inspected using:

```python
df.info()
```

The following information is analyzed:

- Number of observations
- Number of features
- Data types
- Missing values
- Memory usage

This helps determine the preprocessing techniques required for each feature type.

---

### Descriptive Statistics

Statistical summaries are generated using:

```python
df.describe()
```

Key statistics include:

- Mean
- Standard Deviation
- Minimum
- Maximum
- Quartiles
- Median

These statistics help identify abnormal distributions and potential outliers.

---

### Correlation Analysis

A correlation matrix is generated for numerical variables to understand relationships among features.

The correlation heatmap helps identify:

- Highly correlated variables
- Redundant features
- Potential multicollinearity

This information is useful during feature engineering and model interpretation.

---

### Feature Distribution

Histograms are generated for every feature.

Distribution analysis provides insights into:

- Skewed variables
- Uniform distributions
- Multi-modal distributions
- Potential outliers

Both numerical and encoded categorical variables are inspected to ensure proper preprocessing decisions.

---

### Initial Data Quality Assessment

Before preprocessing, the dataset is evaluated for:

- Missing values
- Duplicate observations
- Identifier columns
- Feature consistency
- Data completeness

The results of this assessment determine the preprocessing strategy implemented in the next stage.

---
---

## 💼 Business Impact & Data-Driven Insights

Machine Learning tidak hanya digunakan untuk membangun model, tetapi juga membantu menghasilkan insight yang dapat mendukung pengambilan keputusan berbasis data. Berdasarkan hasil proses clustering dan klasifikasi, beberapa keputusan strategis yang dapat diperoleh antara lain:

### 📍 Segmentasi Data Secara Otomatis
Proses clustering mampu mengelompokkan data berdasarkan karakteristik yang mirip tanpa memerlukan label sebelumnya. Setiap cluster memiliki pola yang berbeda sehingga dapat digunakan sebagai dasar segmentasi data.

Contoh manfaat:

- Mengidentifikasi kelompok pelanggan berdasarkan perilaku transaksi.
- Mengelompokkan tingkat risiko berdasarkan karakteristik tertentu.
- Menemukan pola tersembunyi (hidden pattern) yang sulit diamati secara manual.

---

### 📍 Interpretasi Karakteristik Setiap Cluster

Setelah proses clustering selesai, dilakukan analisis statistik terhadap masing-masing cluster menggunakan nilai:

- Mean
- Minimum
- Maximum
- Mode (fitur kategorikal)

Tahap ini bertujuan untuk menjelaskan karakteristik unik dari setiap kelompok sehingga hasil clustering tidak hanya menghasilkan label, tetapi juga dapat dipahami secara bisnis.

---

### 📍 Transformasi Unsupervised menjadi Supervised Learning

Output clustering kemudian dijadikan label **Target**.

Dengan pendekatan ini:

```
Dataset
        ↓
Preprocessing
        ↓
Clustering
        ↓
Target
        ↓
Classification
```

Pendekatan tersebut memungkinkan pembangunan model klasifikasi yang mampu memprediksi cluster baru secara otomatis tanpa harus menjalankan proses clustering setiap kali terdapat data baru.

---

### 📍 Perbandingan Berbagai Model Klasifikasi

Repositori ini tidak hanya membangun satu model klasifikasi.

Beberapa algoritma dibandingkan menggunakan metrik evaluasi seperti:

- Accuracy
- Precision
- Recall
- F1-Score

Pendekatan ini memberikan gambaran mengenai model mana yang paling sesuai terhadap karakteristik dataset.

---

### 📍 Hyperparameter Optimization

Model terbaik kemudian dioptimalkan menggunakan proses hyperparameter tuning sehingga performanya dapat meningkat dibandingkan model baseline.

Model hasil tuning disimpan sebagai model produksi yang siap digunakan untuk proses inferensi.

---

## 📊 Machine Learning Pipeline

Seluruh proses pada repository ini mengikuti pipeline Machine Learning yang umum digunakan dalam industri.

```text
Dataset
    │
    ▼
Exploratory Data Analysis (EDA)
    │
    ▼
Data Cleaning
    │
    ▼
Feature Engineering
    │
    ▼
Encoding
    │
    ▼
Scaling
    │
    ▼
Feature Binning
    │
    ▼
K-Means Clustering
    │
    ▼
Silhouette Score Evaluation
    │
    ▼
PCA Visualization
    │
    ▼
Cluster Interpretation
    │
    ▼
Inverse Transform
    │
    ▼
Generate Target Label
    │
    ▼
Train-Test Split
    │
    ▼
Decision Tree
    │
    ▼
Additional Classification Models
    │
    ▼
Hyperparameter Tuning
    │
    ▼
Model Evaluation
    │
    ▼
Save Trained Models (.h5)
```

---

## 📦 Generated Artifacts

Selama proses eksperimen, beberapa artefak dihasilkan sebagai output pipeline.

| Artifact | Description |
|-----------|-------------|
| `model_clustering.h5` | Model K-Means Clustering |
| `PCA_model_clustering.h5` | Model PCA untuk visualisasi cluster |
| `decision_tree_model.h5` | Baseline Decision Tree |
| `explore_<algorithm>_classification.h5` | Model klasifikasi tambahan |
| `tuning_classification.h5` | Model hasil hyperparameter tuning |
| `data_clustering.csv` | Dataset preprocessing dengan label Target |
| `data_clustering_inverse.csv` | Dataset hasil inverse transform beserta Target |

---

## 🚀 How to Run

### 1. Clone Repository

```bash
git clone https://github.com/ahmdltf/NAMA_REPOSITORY.git
cd NAMA_REPOSITORY
```

---

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

atau

```bash
pip install pandas numpy matplotlib seaborn scikit-learn yellowbrick joblib
```

---

### 3. Jalankan Notebook Clustering

Notebook ini akan menjalankan seluruh proses:

- Exploratory Data Analysis
- Data Cleaning
- Encoding
- Scaling
- Feature Engineering
- Clustering
- PCA
- Interpretasi Cluster

Output:

```
model_clustering.h5
PCA_model_clustering.h5
data_clustering.csv
data_clustering_inverse.csv
```

---

### 4. Jalankan Notebook Classification

Notebook kedua digunakan untuk:

- Load dataset hasil clustering
- Train-Test Split
- Decision Tree
- Model Comparison
- Hyperparameter Tuning
- Evaluasi Model

Output:

```
decision_tree_model.h5
explore_random_forest_classification.h5
tuning_classification.h5
```

---

## 📚 Libraries

Project ini dibangun menggunakan ekosistem Python Data Science.

| Library | Purpose |
|----------|----------|
| Pandas | Data manipulation |
| NumPy | Numerical computation |
| Matplotlib | Visualization |
| Seaborn | Statistical visualization |
| Scikit-Learn | Machine Learning |
| Yellowbrick | Elbow Method Visualization |
| Joblib | Model Serialization |

---

## 🎯 Learning Outcomes

Melalui proyek ini, saya mengimplementasikan keseluruhan pipeline Machine Learning mulai dari tahap eksplorasi data hingga pembangunan model klasifikasi.

Kompetensi yang diterapkan meliputi:

- Exploratory Data Analysis (EDA)
- Data Cleaning
- Missing Value Handling
- Duplicate Handling
- Feature Engineering
- Label Encoding
- Feature Scaling
- Feature Binning
- Unsupervised Learning (K-Means)
- Cluster Evaluation (Silhouette Score)
- PCA Visualization
- Cluster Interpretation
- Supervised Learning
- Decision Tree Classification
- Model Comparison
- Hyperparameter Optimization
- Model Serialization menggunakan Joblib
- Penyusunan pipeline Machine Learning yang terstruktur dan dapat direproduksi.

---

## 👨‍💻 Author

**Ahmad Latif**

Industrial Engineering Student | Data Scientist | Machine Learning Enthusiast

GitHub: https://github.com/ahmdltf

LinkedIn: https://linkedin.com/in/ahmadlatif

---

## ⭐ Repository Goals

Repository ini dibuat sebagai implementasi end-to-end Machine Learning untuk menunjukkan pemahaman mengenai:

- Data preprocessing
- Unsupervised Learning
- Supervised Learning
- Feature Engineering
- Model Evaluation
- Model Optimization
- Machine Learning Pipeline

dengan pendekatan yang mengikuti praktik umum dalam pengembangan proyek Machine Learning sehingga mudah direproduksi, dipelajari, dan dikembangkan lebih lanjut.
