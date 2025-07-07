import pandas as pd
import joblib
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

print("Memulai proses pelatihan model...")

# Muat dataset
df = pd.read_csv('netflix_customer_churn.csv')

# Preprocessing
X = df.drop(['customer_id', 'churned'], axis=1)
y = df['churned']
categorical_cols = X.select_dtypes(include=['object']).columns
X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

# Simpan kolom untuk digunakan di Streamlit
model_columns = X_encoded.columns
joblib.dump(model_columns, 'model_columns.pkl')
print("Kolom model disimpan.")

# Bagi data
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42, stratify=y)
print("Data telah dibagi.")

# Latih dan simpan Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
joblib.dump(rf_model, 'randomforest_model.pkl')
print("Model Random Forest berhasil dilatih ulang dan disimpan.")

# Latih dan simpan XGBoost
xgb_model = xgb.XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='logloss')
xgb_model.fit(X_train, y_train)
joblib.dump(xgb_model, 'xgboost_model.pkl')
print(" Model XGBoost berhasil dilatih ulang dan disimpan.")
