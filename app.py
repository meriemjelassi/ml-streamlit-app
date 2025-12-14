import streamlit as st
import joblib
import numpy as np

# Charger les objets
scaler = joblib.load("scaler.pkl")
pca = joblib.load("pca.pkl")
model = joblib.load("model.pkl")

st.title("Application ML – Risque de crédit")
st.write("Entrez les informations du client")

# 🔹 VARIABLES RÉELLES (exemples)
age = st.number_input("Âge", min_value=18, max_value=100)
income = st.number_input("Revenu annuel")
loan_amount = st.number_input("Montant du prêt")
loan_percent_income = st.number_input("Pourcentage du revenu")
interest_rate = st.number_input("Taux d’intérêt")
credit_history = st.number_input("Historique de crédit")
employment_years = st.number_input("Années d'emploi")
home_ownership = st.number_input("Type de logement (encodé)")
loan_intent = st.number_input("Intention du prêt (encodé)")
grade = st.number_input("Grade du prêt")
default_on_file = st.number_input("Défaut antérieur (0/1)")

if st.button("Prédire le risque"):
    # 1️⃣ données brutes
    X = np.array([[age, income, loan_amount, loan_percent_income,
                   interest_rate, credit_history, employment_years,
                   home_ownership, loan_intent, grade, default_on_file]])

    # 2️⃣ scaler
    X_scaled = scaler.transform(X)

    # 3️⃣ PCA
    X_pca = pca.transform(X_scaled)

    # 4️⃣ prédiction
    prediction = model.predict(X_pca)

    st.success(f"Résultat du modèle : {prediction[0]}")
