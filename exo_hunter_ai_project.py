# This is a cleaned, ready-to-run ExoHunter AI project without folder diagrams or invalid characters.

# preprocess.py
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

def load_and_preprocess(file_path):
    df = pd.read_csv(file_path)
    df = df.dropna(subset=['pl_disposition'])
    df['pl_orbper'] = df['pl_orbper'].fillna(df['pl_orbper'].median())
    df['pl_rade'] = df['pl_rade'].fillna(df['pl_rade'].median())
    df['pl_bmassj'] = df['pl_bmassj'].fillna(df['pl_bmassj'].median())
    df['st_teff'] = df['st_teff'].fillna(df['st_teff'].median())
    df['st_rad'] = df['st_rad'].fillna(df['st_rad'].median())
    df['st_mass'] = df['st_mass'].fillna(df['st_mass'].median())
    le = LabelEncoder()
    df['target'] = le.fit_transform(df['pl_disposition'])
    X = df[['pl_orbper', 'pl_rade', 'pl_bmassj', 'st_teff', 'st_rad', 'st_mass']]
    y = df['target']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y, scaler, le


# train_model.py
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from preprocess import load_and_preprocess

X, y, scaler, le = load_and_preprocess('data/exoplanet_data.csv')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=le.classes_))
with open('models/exo_model.pkl', 'wb') as f:
    pickle.dump({'model': model, 'scaler': scaler, 'label_encoder': le}, f)
print("Model saved successfully!")


# app.py
import streamlit as st
import pickle
import numpy as np

with open('models/exo_model.pkl', 'rb') as f:
    data = pickle.load(f)
model = data['model']
scaler = data['scaler']
le = data['label_encoder']

st.title("ExoHunter AI 🌌")
st.write("Predict if a potential exoplanet is confirmed, candidate, or false positive.")
orbper = st.number_input("Orbital Period (days)", min_value=0.1, value=10.0)
rade = st.number_input("Planet Radius (Earth radii)", min_value=0.1, value=1.0)
bmassj = st.number_input("Planet Mass (Jupiter masses)", min_value=0.0, value=0.5)
teff = st.number_input("Star Temperature (K)", min_value=1000, value=5500)
srad = st.number_input("Star Radius (Solar radii)", min_value=0.1, value=1.0)
smass = st.number_input("Star Mass (Solar masses)", min_value=0.1, value=1.0)

if st.button("Predict"):
    features = np.array([[orbper, rade, bmassj, teff, srad, smass]])
    features_scaled = scaler.transform(features)
    pred = model.predict(features_scaled)
    pred_label = le.inverse_transform(pred)[0]
    st.success(f"Prediction: **{pred_label}**")


# requirements.txt
pandas
numpy
scikit-learn
streamlit


# README.md
# ExoHunter AI


```bash
pip install -r requirements.txt
```

```bash
python train_model.py
```
streamlit run app.py
```

