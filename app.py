import streamlit as st
import joblib
import numpy as np

# Charger les objets entraînés
scaler = joblib.load("scaler.pkl")
pca = joblib.load("pca.pkl")
model = joblib.load("model.pkl")

st.title("Application ML – Risque de Crédit")

st.write("Veuillez entrer les informations du client")

# ⚠️ Ces variables DOIVENT correspondre à celles du notebook
age = st.number_input("Âge", min_value=18, max_value=100)
income = st.number_input("Revenu annuel")
loan_amnt = st.number_input("Montant du prêt")
loan_percent_income = st.number_input("Pourcentage du revenu dédié au prêt")
credit_history = st.number_input("Ancienneté du crédit (années)")
interest_rate = st.number_input("Taux d’intérêt")

# ➕ compléter pour arriver EXACTEMENT au nombre de variables originales
# Exemple fictif :
var7 = st.number_input("Variable 7")
var8 = st.number_input("Variable 8")
var9 = st.number_input("Variable 9")
var10 = st.number_input("Variable 10")
var11 = st.number_input("Variable 11")

if st.button("Prédire le risque"):
    # 1️⃣ données brutes (11 variables)
    X = np.array([[age, income, loan_amnt, loan_percent_income,
                   credit_history, interest_rate,
                   var7, var8, var9, var10, var11]])

    # 2️⃣ standardisation
    X_scaled = scaler.transform(X)

    # 3️⃣ PCA
    X_pca = pca.transform(X_scaled)

    # 4️⃣ prédiction
    prediction = model.predict(X_pca)

    if prediction[0] == 1:
        st.error("⚠️ Client à risque de défaut")
    else:
        st.success("✅ Client solvable")
