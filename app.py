import streamlit as st
import pandas as pd
import joblib

@st.cache_data
def load_assets():
    df_for_inputs = pd.read_csv('netflix_customer_churn.csv')
    rf_model = joblib.load('randomforest_model.pkl')
    xgb_model = joblib.load('xgboost_model.pkl')
    model_columns = joblib.load('model_columns.pkl')
    return df_for_inputs, rf_model, xgb_model, model_columns

def get_prediction(input_df, model, columns):
    input_processed = pd.get_dummies(input_df)
    input_final = input_processed.reindex(columns=columns, fill_value=0)
    prediction = model.predict(input_final)[0]
    probability = model.predict_proba(input_final)[0]
    return prediction, probability

df, rf_model, xgb_model, model_columns = load_assets()

st.set_page_config(page_title="Prediksi Churn Netflix", page_icon="📺")
st.title('Prediksi Churn Pelanggan Netflix 📺')

st.sidebar.header('Input Data Pelanggan')
model_selection = st.sidebar.selectbox("Pilih Model Prediksi", ["XGBoost", "Random Forest"])
st.sidebar.markdown("---")

with st.sidebar.form(key='input_form'):
    age = st.slider('Umur', 18, 70, 35)
    gender = st.selectbox('Gender', df['gender'].unique())
    subscription_type = st.selectbox('Tipe Langganan', df['subscription_type'].unique())
    watch_hours = st.slider('Jam Menonton (Bulan Terakhir)', 0.0, 500.0, 150.0)
    last_login_days = st.slider('Hari Sejak Login Terakhir', 0, 90, 10)
    monthly_fee = st.slider('Biaya Bulanan ($)', 5.0, 25.0, 15.0)
    number_of_profiles = st.slider('Jumlah Profil', 1, 10, 3)
    avg_watch_time_per_day = st.slider('Rata-rata Menonton per Hari (menit)', 30.0, 240.0, 120.0)
    favorite_genre = st.selectbox('Genre Favorit', df['favorite_genre'].unique())
    payment_method = st.selectbox('Metode Pembayaran', df['payment_method'].unique())
    device = st.selectbox('Device', df['device'].unique())
    region = st.selectbox('Region', df['region'].unique())
    
    submit_button = st.form_submit_button(label='Lakukan Prediksi')

if submit_button:
    input_data = {
        'age': age,
        'gender': gender,
        'subscription_type': subscription_type,
        'watch_hours': watch_hours,
        'last_login_days': last_login_days,
        'region': region,
        'device': device,
        'monthly_fee': monthly_fee,
        'payment_method': payment_method,
        'number_of_profiles': number_of_profiles,
        'avg_watch_time_per_day': avg_watch_time_per_day,
        'favorite_genre': favorite_genre
    }
    input_df = pd.DataFrame([input_data])

    model_to_use = xgb_model if model_selection == "XGBoost" else rf_model
    pred, proba = get_prediction(input_df, model_to_use, model_columns)

    st.subheader(f'Hasil Prediksi dengan Model {model_selection}')
    
    if pred == 1:
        st.error(f"Pelanggan ini **BERPOTENSI CHURN**")
        st.metric(label="Probabilitas Churn", value=f"{proba[1]:.2%}")
    else:
        st.success(f"Pelanggan ini **CENDERUNG BERTAHAN**")
        st.metric(label="Probabilitas Bertahan", value=f"{proba[0]:.2%}")
else:
    st.info('Silakan isi data pelanggan di sidebar kiri dan klik tombol "Lakukan Prediksi".')