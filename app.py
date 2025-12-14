import streamlit as st
import pandas as pd
import joblib

# Charger les modèles pré-entraînés
clf_model = joblib.load('gradient_boosting_classifier.pkl')
reg_model = joblib.load('random_forest_regressor.pkl')
scaler = joblib.load('scaler.pkl')
pca = joblib.load('pca_model.pkl')

st.title("📊 Assistant de Prédiction de Risque de Crédit")

st.sidebar.header("📝 Informations du client")

# Formulaire de saisie
age = st.sidebar.number_input("Âge", min_value=18, max_value=100)
income = st.sidebar.number_input("Revenu annuel ($)", min_value=0)
loan_amount = st.sidebar.number_input("Montant du prêt demandé ($)", min_value=0)
loan_intent = st.sidebar.selectbox("Intention du prêt", ["Personnel", "Éducation", "Médical", "Entreprise", "Amélioration", "Dette"])
home_ownership = st.sidebar.selectbox("Type de logement", ["Locataire", "Propriétaire", "Hypothèque", "Autre"])
credit_score = st.sidebar.slider("Score de crédit", 300, 850, 650)

# Bouton de prédiction
if st.sidebar.button("🔍 Analyser le risque"):

    # Préparer les données
    input_data = pd.DataFrame({
        'age': [age],
        'income': [income],
        'loan_amount': [loan_amount],
        'loan_intent': [loan_intent],
        'home_ownership': [home_ownership],
        'credit_score': [credit_score]
        # ... autres variables
    })

    # Transformation
    input_scaled = scaler.transform(input_data)
    input_pca = pca.transform(input_scaled)

    # Prédiction
    risk_prediction = clf_model.predict(input_pca)[0]
    loan_recommendation = reg_model.predict(input_pca)[0]

    # Affichage
    st.subheader("📈 Résultats de l'analyse")

    if risk_prediction == 0:
        st.success("✅ **Risque faible** – Client recommandé pour approbation.")
        st.metric("Montant recommandé", f"{loan_recommendation:,.2f} $")
    else:
        st.error("❌ **Risque élevé** – Défaut probable.")
        st.warning("Montant recommandé : 0 $ (refus recommandé)")

    # Explication
    with st.expander("📊 Détails techniques"):
        st.write("**Modèle utilisé :** Gradient Boosting (F1-Score = 0.825)")
        st.write("**Fiabilité estimée :** 93,4 %")
        st.write("**Variables clés :** Revenu, Score de crédit, Intention du prêt")
