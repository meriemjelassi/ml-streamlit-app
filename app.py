import streamlit as st
import joblib
import numpy as np

# Charger les objets
scaler = joblib.load("scaler.pkl")
pca = joblib.load("pca.pkl")
model = joblib.load("model.pkl")

st.title("Application ML – Risque de crédit")

st.write("Veuillez entrer les informations du client")

# 🔹 Entrées utilisateur (variables réelles)
person_age = st.number_input("Âge du client", min_value=18, max_value=100)
person_income = st.number_input("Revenu annuel")
loan_amnt = st.number_input("Montant du prêt")
loan_percent_income = st.number_input("Pourcentage du revenu dédié au prêt")
credit_history_length = st.number_input("Ancienneté du crédit (années)")
interest_rate = st.number_input("Taux d'intérêt (%)")

# Bouton
if st.button("Prédire le risque"):
    # 1️⃣ Mettre les données dans le bon format
    X = np.array([[person_age,
                   person_income,
                   loan_amnt,
                   loan_percent_income,
                   credit_history_length,
                   interest_rate]])

    # 2️⃣ Standardisation
    X_scaled = scaler.transform(X)

    # 3️⃣ PCA
    X_pca = pca.transform(X_scaled)

    # 4️⃣ Prédiction
    prediction = model.predict(X_pca)

    if prediction[0] == 1:
        st.error("⚠️ Client à RISQUE de défaut")
    else:
        st.success("✅ Client à FAIBLE risque")
