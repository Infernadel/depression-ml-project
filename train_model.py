# ===============================================================
# 1. IMPORT LIBRARIES
# ===============================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error
)

# ===============================================================
# 2. PENGUMPULAN & PERSIAPAN DATA
# ===============================================================

df = pd.read_csv("final_depression_dataset_1.csv")

# Standarisasi nama kolom
def standardize_column_names(df):
    new_cols = []
    for col in df.columns:
        standardized = col.lower().replace(' ', '_').replace('/', '_').replace('?', '').replace('-', '_').strip()
        new_cols.append(standardized)
    df.columns = new_cols
    return df

df = standardize_column_names(df)

# Hapus kolom tak relevan
df = df.drop(columns=['name', 'cgpa', 'city'], errors='ignore')

df.info()

# ===============================================================
# 3. PREPROCESSING — PENGGABUNGAN FITUR
# ===============================================================

# --- 3.1 Gabungkan profesi ---
df['combined_profession'] = df['working_professional_or_student']
df.loc[df['working_professional_or_student'] == 'Working Professional', 'combined_profession'] = df['profession']
df = df.drop(columns=['working_professional_or_student', 'profession'])

# TIDAK ADA THRESHOLD – biarkan kategori apa adanya
df['combined_profession'] = df['combined_profession'].fillna('Unknown Profession')

# --- 3.2 Degree ---
df['degree'] = df['degree'].fillna('Unknown Degree')

# TIDAK ADA THRESHOLD – degree tidak digabung

# --- 3.3 Gabung fitur Pressure ---
for col in ['academic_pressure', 'work_pressure']:
    if col in df.columns:
        df[col] = df[col].fillna(0).astype(int)

df['pressure_score'] = df['academic_pressure'] + df['work_pressure']
df = df.drop(columns=['academic_pressure', 'work_pressure'])

# --- 3.4 Gabung fitur Satisfaction ---
for col in ['study_satisfaction', 'job_satisfaction']:
    if col in df.columns and df[col].dtype != object:
        df[col] = df[col].fillna(df[col].median())
    else:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(df[col].median())

df['overall_satisfaction'] = (df['study_satisfaction'] + df['job_satisfaction']) / 2
df = df.drop(columns=['study_satisfaction', 'job_satisfaction'])

df.info()

# ===============================================================
# 4. CEK MISSING VALUE
# ===============================================================
print("\nMissing Value per Kolom:")
print(df.isnull().sum())

# ===============================================================
# 5. DETEKSI & PENGHAPUSAN OUTLIER
# ===============================================================

print("\n===========================")
print("🔍 DETEKSI OUTLIER (IQR)")
print("===========================")

initial_rows = df.shape[0]
print(f"Jumlah data awal: {initial_rows}")

numeric_features = ['age', 'work_study_hours', 'financial_stress', 'pressure_score', 'overall_satisfaction']

for col in numeric_features:
    if col in df.columns:
        Q1, Q3 = df[col].quantile([0.25, 0.75])
        IQR = Q3 - Q1

        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        before = df.shape[0]
        df = df[(df[col] >= lower) & (df[col] <= upper)]
        after = df.shape[0]

        print(f"- {col}: Terhapus {before - after} outlier")

print(f"Jumlah data akhir: {df.shape[0]}")
print("===========================\n")

# ===============================================================
# 6. PEMISAHAN FITUR X & y
# ===============================================================

y = df['depression']
X = df.drop(columns=['depression'])

categorical_features = [
    'gender', 'combined_profession', 'degree',
    'sleep_duration', 'dietary_habits',
    'have_you_ever_had_suicidal_thoughts_',
    'family_history_of_mental_illness'
]

# ===============================================================
# 7. PIPELINE PREPROCESSING
# ===============================================================

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

X_ready = preprocessor.fit_transform(X)
y_ready = y.apply(lambda x: 1 if x == "Yes" else 0)

# ===============================================================
# 8. TRAIN–TEST SPLIT
# ===============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X_ready, y_ready,
    test_size=0.2,
    random_state=42,
    stratify=y_ready
)

print(f"X_train: {X_train.shape}, X_test: {X_test.shape}")

# ===============================================================
# 9. TRAIN MODEL
# ===============================================================

model_lr = LogisticRegression(random_state=42, solver='liblinear')
model_lr.fit(X_train, y_train)

# ===============================================================
# 10. PREDIKSI & EVALUASI
# ===============================================================

y_pred = model_lr.predict(X_test)

print("\n=== Evaluasi Model Logistic Regression ===")
print(classification_report(y_test, y_pred))
print(f"Akurasi: {accuracy_score(y_test, y_pred):.4f}")

accuracy  = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall    = recall_score(y_test, y_pred)
f1        = f1_score(y_test, y_pred)

mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print("\n========================")
print("EVALUASI MODEL")
print("========================")
print(f"Akurasi  : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-Score : {f1:.4f}")
print(f"MSE      : {mse:.4f}")
print(f"MAE      : {mae:.4f}")

print("\n========================")
print("ANALISIS MODEL")
print("========================")

if accuracy > 0.80:
    print("✔ Model memiliki akurasi yang sangat baik.")
elif accuracy > 0.65:
    print("✔ Model cukup baik, tetapi masih bisa ditingkatkan.")
else:
    print("⚠ Model kurang akurat, disarankan perbaikan preprocessing atau algoritma lain.")

if precision > recall:
    print("✔ Precision lebih tinggi → Model lebih ketat, minim false positive.")
else:
    print("⚠ Recall lebih tinggi → Model lebih sensitif, false positive mungkin lebih besar.")

if recall > 0.75:
    print("✔ Recall tinggi → Model baik mendeteksi depresi.")
else:
    print("⚠ Recall rendah → Model bisa gagal mendeteksi depresi.")

print(f"✔ F1-Score ({f1:.4f}) seimbang antara precision & recall.")
print(f"✔ MAE ({mae:.4f}) adalah rata-rata kesalahan prediksi.")
print(f"✔ MSE ({mse:.4f}) menggambarkan error keseluruhan.\n")

# Confusion Matrix
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)
print("\n")

plt.figure(figsize=(10,8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Pred: Tidak', 'Pred: Depresi'],
            yticklabels=['Actual: Tidak', 'Actual: Depresi'])
plt.title("Confusion Matrix - Logistic Regression")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()



import pickle
import os

print("\n" + "="*60)
print("💾 MENYIMPAN MODEL")
print("="*60)

# Simpan preprocessor dan model yang sudah di-fit
model_data = {
    'preprocessor': preprocessor,  # dari script training
    'model': model_lr,              # dari script training
    'numeric_features': ['age', 'work_study_hours', 'financial_stress', 
                         'pressure_score', 'overall_satisfaction'],
    'categorical_features': ['gender', 'combined_profession', 'degree',
                            'sleep_duration', 'dietary_habits',
                            'have_you_ever_had_suicidal_thoughts_',
                            'family_history_of_mental_illness']
}

filename = 'depression_model.pkl'
print(f"⏳ Menyimpan ke {filename}...")

with open(filename, 'wb') as f:
    pickle.dump(model_data, f)

file_size = os.path.getsize(filename) / 1024
print(f"✅ Model berhasil disimpan!")
print(f"   📁 File: {filename}")
print(f"   📊 Ukuran: {file_size:.2f} KB")

# Test prediksi cepat
print("\n⏳ Test prediksi...")
test_data = pd.DataFrame([{
    'age': 25,
    'gender': 'Male',
    'combined_profession': 'Student',
    'degree': 'B.Tech',
    'work_study_hours': 8.0,
    'financial_stress': 3,
    'pressure_score': 5,
    'overall_satisfaction': 3.0,
    'sleep_duration': '7-8 hours',
    'dietary_habits': 'Moderate',
    'have_you_ever_had_suicidal_thoughts_': 'No',
    'family_history_of_mental_illness': 'No'
}])

X_test_transformed = preprocessor.transform(test_data)
pred = model_lr.predict(X_test_transformed)
proba = model_lr.predict_proba(X_test_transformed)

print(f"✅ Prediksi: {'Depressed' if pred[0] == 1 else 'Not Depressed'}")
print(f"✅ Probabilitas: {proba[0][1]*100:.2f}%")

print("\n" + "="*60)
print("✅ SELESAI! Model siap untuk aplikasi Streamlit")
print("="*60)
print("Jalankan: streamlit run app.py")
print("="*60)