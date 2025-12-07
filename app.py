import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

# ===============================================================
# KONFIGURASI HALAMAN
# ===============================================================
st.set_page_config(
    page_title="Depression Prediction App",
    page_icon="🧠",
    layout="wide"
)

# ===============================================================
# FUNGSI UNTUK MEMBUAT & MENYIMPAN MODEL
# ===============================================================
def create_and_save_model():
    """
    Fungsi ini membuat model dummy untuk keperluan demonstrasi.
    Dalam praktik nyata, Anda akan melatih model dengan data sesungguhnya.
    """
    # Definisi fitur
    numeric_features = ['work_study_hours', 'financial_stress', 
                       'pressure_score', 'overall_satisfaction']
    categorical_features = ['gender', 'combined_profession', 'degree',
                          'sleep_duration', 'dietary_habits',
                          'have_you_ever_had_suicidal_thoughts_',
                          'family_history_of_mental_illness']
    
    # Pipeline preprocessing
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
    
    # Model
    model = LogisticRegression(random_state=42, solver='liblinear')
    
    # Pipeline lengkap
    full_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])
    
    return full_pipeline, numeric_features, categorical_features

# ===============================================================
# FUNGSI LOAD MODEL
# ===============================================================
@st.cache_resource
def load_model():
    """
    Memuat model dari file pickle.
    Jika file tidak ada, buat model dummy untuk demo.
    """
    try:
        with open('depression_model.pkl', 'rb') as f:
            model_data = pickle.load(f)
        return model_data
    except FileNotFoundError:
        st.warning("⚠️ Model file tidak ditemukan. Menggunakan model dummy untuk demonstrasi.")
        model_data, _, _ = create_and_save_model()
        return {
            'preprocessor': model_data.named_steps['preprocessor'],
            'model': model_data.named_steps['classifier'],
            'numeric_features': ['work_study_hours', 'financial_stress', 
                               'pressure_score', 'overall_satisfaction'],
            'categorical_features': ['gender', 'combined_profession', 'degree',
                                   'sleep_duration', 'dietary_habits',
                                   'have_you_ever_had_suicidal_thoughts_',
                                   'family_history_of_mental_illness']
        }

# ===============================================================
# FUNGSI PREDIKSI
# ===============================================================
def predict_single(input_dict, model_data):
    """
    Memprediksi satu sampel data.
    
    Parameters:
    - input_dict: Dictionary berisi input user
    - model_data: Dictionary berisi preprocessor dan model
    
    Returns:
    - prediction: 0 (Not Depressed) atau 1 (Depressed)
    - probability: Probabilitas kelas positif (Depressed)
    """
    # Konversi ke DataFrame
    df_input = pd.DataFrame([input_dict])
    
    # Prediksi
    try:
        # Transform data dengan preprocessor
        X_transformed = model_data['preprocessor'].transform(df_input)
        
        # Prediksi dengan model
        prediction = model_data['model'].predict(X_transformed)[0]
        probability = model_data['model'].predict_proba(X_transformed)[0]
        
        return prediction, probability
    except Exception as e:
        st.error(f"❌ Error saat prediksi: {str(e)}")
        return None, None

# ===============================================================
# DATA KATEGORI AKTUAL (dari dataset)
# ===============================================================
CATEGORY_OPTIONS = {
    'gender': ['Male', 'Female'],
    'combined_profession': [
        'Student',
        'Teacher',
        'Doctor',
        'Lawyer',
        'Accountant',
        'Software Engineer',
        'Mechanical Engineer',
        'Civil Engineer',
        'Architect',
        'Manager',
        'HR Manager',
        'Marketing Manager',
        'Consultant',
        'Business Analyst',
        'Financial Analyst',
        'Finanancial Analyst',  # typo dari dataset asli
        'Data Scientist',
        'Researcher',
        'Research Analyst',
        'Investment Banker',
        'Entrepreneur',
        'Sales Executive',
        'Customer Support',
        'Content Writer',
        'Digital Marketer',
        'UX/UI Designer',
        'Graphic Designer',
        'Pharmacist',
        'Chemist',
        'Chef',
        'Pilot',
        'Judge',
        'Plumber',
        'Electrician',
        'Travel Consultant',
        'Educational Consultant',
        'Unknown Profession'  # untuk handle nan
    ],
    'degree': [
        'B.Tech',
        'BE',
        'B.Arch',
        'B.Pharm',
        'B.Ed',
        'BBA',
        'BHM',
        'BCA',
        'BA',
        'B.Com',
        'BSc',
        'M.Tech',
        'ME',
        'MBA',
        'MHM',
        'MCA',
        'M.Ed',
        'MA',
        'M.Com',
        'M.Pharm',
        'MSc',
        'LLB',
        'LLM',
        'MBBS',
        'MD',
        'PhD',
        'Class 12'
    ],
    'sleep_duration': [
        'Less than 5 hours', '5-6 hours', '7-8 hours', 'More than 8 hours'
    ],
    'dietary_habits': [
        'Unhealthy', 'Moderate', 'Healthy'
    ],
    'have_you_ever_had_suicidal_thoughts_': ['No', 'Yes'],
    'family_history_of_mental_illness': ['No', 'Yes']
}

# ===============================================================
# MAIN APP
# ===============================================================
def main():
    # Header
    st.title("🧠 Depression Prediction Application")
    st.markdown("---")
    
    # Deskripsi
    st.markdown("""
    ### 📋 Tentang Aplikasi
    Aplikasi ini menggunakan **Machine Learning (Logistic Regression)** untuk memprediksi 
    kemungkinan seseorang mengalami depresi berdasarkan gaya hidup dan latar belakang mereka.
    
    **Cara Menggunakan:**
    1. Isi semua informasi yang diminta di formulir di bawah
    2. Klik tombol **"🔮 Predict Depression"**
    3. Lihat hasil prediksi dan probabilitasnya
    
    ⚠️ **Disclaimer:** Aplikasi ini hanya untuk tujuan edukasi dan skrining awal. 
    Untuk diagnosis yang akurat, konsultasikan dengan profesional kesehatan mental.
    """)
    
    st.markdown("---")
    
    # Load model
    model_data = load_model()
    
    # Buat form input
    st.header("📝 Input Data")
    
    # Buat 2 kolom untuk layout yang lebih rapi
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("👤 Informasi Demografis")
        
        
        # Gender
        st.markdown("**Gender**")
        st.caption("Pilih jenis kelamin Anda.")
        gender = st.selectbox(
            "Gender",
            options=CATEGORY_OPTIONS['gender'],
            label_visibility="collapsed"
        )
        
        # Combined Profession
        st.markdown("**Profession**")
        st.caption("Pilih profesi atau status pekerjaan Anda saat ini. Pilih 'Student' jika Anda masih bersekolah/kuliah, atau 'Unknown Profession' jika tidak ada yang sesuai.")
        combined_profession = st.selectbox(
            "Combined Profession",
            options=CATEGORY_OPTIONS['combined_profession'],
            index=0,  # default ke 'Student'
            label_visibility="collapsed"
        )
        
        # Degree
        st.markdown("**Degree**")
        st.caption("Pilih tingkat pendidikan tertinggi Anda. Contoh: B.Tech untuk Sarjana Teknik, MBA untuk Master Bisnis. Untuk lulusan SMA pilih Class 12")
        degree = st.selectbox(
            "Degree",
            options=CATEGORY_OPTIONS['degree'],
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        st.subheader("🏥 Riwayat Kesehatan Mental")
        
        # Suicidal Thoughts
        st.markdown("**Have You Ever Had Suicidal Thoughts?**")
        st.caption("Apakah Anda pernah memiliki pikiran untuk mengakhiri hidup? Jawab dengan jujur untuk hasil yang akurat.")
        suicidal_thoughts = st.selectbox(
            "Suicidal Thoughts",
            options=CATEGORY_OPTIONS['have_you_ever_had_suicidal_thoughts_'],
            label_visibility="collapsed"
        )
        
        # Family History
        st.markdown("**Family History of Mental Illness**")
        st.caption("Apakah ada anggota keluarga yang memiliki riwayat gangguan mental (depresi, kecemasan, dll)?")
        family_history = st.selectbox(
            "Family History",
            options=CATEGORY_OPTIONS['family_history_of_mental_illness'],
            label_visibility="collapsed"
        )
    
    with col2:
        st.subheader("💼 Aktivitas & Tekanan")
        
        # Work/Study Hours
        st.markdown("**Work/Study Hours**")
        st.caption("Berapa jam per hari Anda bekerja atau belajar? Rentang umum: 0-16 jam.")
        work_study_hours = st.number_input(
            "Work/Study Hours",
            min_value=0.0,
            max_value=24.0,
            value=8.0,
            step=0.5,
            label_visibility="collapsed"
        )
        
        # Financial Stress
        st.markdown("**Financial Stress**")
        st.caption("Seberapa besar tingkat stres finansial Anda? Skala 1-5, di mana 1 = tidak stres, 5 = sangat stres.")
        financial_stress = st.slider(
            "Financial Stress",
            min_value=1.0,
            max_value=5.0,
            value=3.0,
            step=0.01,
            label_visibility="collapsed"
        )
        
        # Pressure Score
        st.markdown("**Pressure Score**")
        st.caption("Skor total tekanan yang Anda rasakan (akademik + pekerjaan). Skala 1-5, di mana 1 = tidak ada tekanan, 5 = tekanan sangat tinggi.")
        pressure_score = st.slider(
            "Pressure Score",
            min_value=1.0,
            max_value=5.0,
            value=3.0,
            step=0.01,
            label_visibility="collapsed"
        )
        
        # Overall Satisfaction
        st.markdown("**Overall Satisfaction**")
        st.caption("Seberapa puas Anda dengan kehidupan kerja/studi Anda? Skala 1-5, di mana 1 = sangat tidak puas, 5 = sangat puas.")
        overall_satisfaction = st.slider(
            "Overall Satisfaction",
            min_value=1.0,
            max_value=5.0,
            value=3.0,
            step=0.01,
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        st.subheader("🌙 Gaya Hidup")
        
        # Sleep Duration
        st.markdown("**Sleep Duration**")
        st.caption("Berapa lama rata-rata Anda tidur setiap malam?")
        sleep_duration = st.selectbox(
            "Sleep Duration",
            options=CATEGORY_OPTIONS['sleep_duration'],
            label_visibility="collapsed"
        )
        
        # Dietary Habits
        st.markdown("**Dietary Habits**")
        st.caption("Bagaimana pola makan Anda secara umum? Healthy = seimbang dan teratur, Moderate = cukup baik, Unhealthy = tidak teratur/junk food.")
        dietary_habits = st.selectbox(
            "Dietary Habits",
            options=CATEGORY_OPTIONS['dietary_habits'],
            label_visibility="collapsed"
        )
    
    st.markdown("---")
    
    # Tombol Predict
    col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])
    with col_btn2:
        predict_button = st.button("🔮 Predict Depression", use_container_width=True, type="primary")
    
    # Proses prediksi
    if predict_button:

        
        if work_study_hours < 0 or work_study_hours > 24:
            st.error("❌ Jam kerja/belajar harus antara 0-24 jam!")
            return
        
        # Siapkan input dictionary
        input_data = {
            'gender': gender,
            'combined_profession': combined_profession,
            'degree': degree,
            'work_study_hours': work_study_hours,
            'financial_stress': financial_stress,
            'pressure_score': pressure_score,
            'overall_satisfaction': overall_satisfaction,
            'sleep_duration': sleep_duration,
            'dietary_habits': dietary_habits,
            'have_you_ever_had_suicidal_thoughts_': suicidal_thoughts,
            'family_history_of_mental_illness': family_history
        }
        
        # Prediksi
        with st.spinner('🔄 Memproses prediksi...'):
            prediction, probability = predict_single(input_data, model_data)
        
        if prediction is not None:
            st.markdown("---")
            st.header("📊 Hasil Prediksi")
            
            # Tampilkan hasil
            col_result1, col_result2 = st.columns(2)
            
            with col_result1:
                if prediction == 1:
                    st.error("### 🔴 Depressed")
                    st.markdown("""
                    Berdasarkan data yang Anda masukkan, model memprediksi Anda 
                    **kemungkinan mengalami depresi**.
                    """)
                else:
                    st.success("### 🟢 Not Depressed")
                    st.markdown("""
                    Berdasarkan data yang Anda masukkan, model memprediksi Anda 
                    **tidak mengalami depresi**.
                    """)
            
            with col_result2:
                st.metric(
                    "Probabilitas Depressed",
                    f"{probability[1]*100:.2f}%"
                )
                st.metric(
                    "Probabilitas Not Depressed",
                    f"{probability[0]*100:.2f}%"
                )
            
            # Progress bar untuk visualisasi probabilitas
            st.markdown("### 📈 Visualisasi Probabilitas")
            st.progress(probability[1])
            
            # Rekomendasi
            st.markdown("---")
            st.markdown("### 💡 Rekomendasi")
            
            if prediction == 1:
                st.warning("""
                **Saran untuk Anda:**
                - 🗣️ Pertimbangkan untuk berbicara dengan profesional kesehatan mental
                - 👨‍👩‍👧‍👦 Ceritakan perasaan Anda kepada orang terdekat
                - 🧘‍♀️ Cobalah teknik relaksasi seperti meditasi atau yoga
                - 😴 Perbaiki pola tidur Anda
                - 🥗 Perhatikan pola makan yang sehat
                - 💪 Lakukan aktivitas fisik secara teratur
                
                **Hotline Kesehatan Mental Indonesia:**
                - **Sehatmental.id** (Kemenkes): 119 ext. 8
                - **Into The Light**: 021-7884-5855
                - **LSM Jangan Bunuh Diri**: 021-9696-9293
                """)
            else:
                st.info("""
                **Pertahankan Kesehatan Mental Anda:**
                - ✅ Terus jaga pola hidup sehat
                - ✅ Pertahankan work-life balance yang baik
                - ✅ Tetap terhubung dengan orang-orang terdekat
                - ✅ Lakukan hobi yang Anda sukai
                - ✅ Istirahat yang cukup
                - ✅ Olahraga teratur
                """)
            
            # Disclaimer
            st.markdown("---")
            st.info("""
            ⚠️ **Penting:** Hasil prediksi ini bukan diagnosis medis. Aplikasi ini hanya 
            memberikan estimasi berdasarkan model machine learning. Untuk diagnosis yang 
            akurat dan penanganan yang tepat, silakan konsultasikan dengan psikolog, 
            psikiater, atau profesional kesehatan mental lainnya.
            """)

# ===============================================================
# INSTRUKSI SAVE MODEL (untuk dijalankan terpisah)
# ===============================================================
def save_trained_model(model, numeric_features, categorical_features, filename='depression_model.pkl'):
    """
    Fungsi untuk menyimpan model yang sudah dilatih.
    Panggil fungsi ini setelah melatih model dengan script training Anda.
    
    Contoh penggunaan setelah training:
    
    from sklearn.pipeline import Pipeline
    
    # Setelah model_lr dan preprocessor siap
    full_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', model_lr)
    ])
    
    # Fit pipeline dengan data training
    full_pipeline.fit(X, y)
    
    # Simpan model
    model_data = {
        'model': full_pipeline,
        'numeric_features': numeric_features,
        'categorical_features': categorical_features
    }
    
    with open('depression_model.pkl', 'wb') as f:
        pickle.dump(model_data, f)
    """
    model_data = {
        'model': model,
        'numeric_features': numeric_features,
        'categorical_features': categorical_features
    }
    
    with open(filename, 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"✅ Model berhasil disimpan ke {filename}")

# ===============================================================
# RUN APP
# ===============================================================
if __name__ == "__main__":
    main()