#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Page de prédiction individuelle
"""

import streamlit as st
import pandas as pd
from datetime import datetime
from ui.components import page_header, warning_message, risk_level_card

@st.cache_data
def get_equipment_codes(valid_equipments):
    """
    Cache pour obtenir la liste des codes d'équipements
    """
    if valid_equipments is not None:
        return sorted(valid_equipments)
    return []

@st.cache_data
def compute_individual_prediction_cached(equipment_code, current_date_str, model_hash, data_hash):
    """
    Cache intelligent pour les prédictions individuelles
    """
    dashboard = st.session_state.dashboard
    
    # Vérifier si on a déjà cette prédiction en cache
    cache_key = f"individual_prediction_{equipment_code}_{model_hash}_{data_hash}_{current_date_str}"
    
    if 'individual_predictions_cache' not in st.session_state:
        st.session_state.individual_predictions_cache = {}
    
    if cache_key in st.session_state.individual_predictions_cache:
        return st.session_state.individual_predictions_cache[cache_key]
    
    # Calculer la prédiction
    prediction = dashboard.predict_failure(equipment_code, current_date_str)
    
    # Stocker en cache
    st.session_state.individual_predictions_cache[cache_key] = prediction
    
    return prediction

def single_equipment_prediction_page():
    """
    Page de prédiction pour un équipement spécifique avec cache intelligent
    """
    page_header("Prédiction Individuelle", "Vue d'ensemble de prédiction de panne pour un équipement valide")
    
    # Vérifier si le modèle et les données sont chargés
    dashboard = st.session_state.dashboard
    model_loaded = dashboard.model_package is not None
    data_loaded = dashboard.df_clean is not None
    
    if not model_loaded or not data_loaded:
        warning_message("Veuillez charger le modèle et les données dans la barre latérale.")
        return
    
    # Vérifier qu'il y a des équipements valides
    if dashboard.valid_equipments is None or len(dashboard.valid_equipments) == 0:
        st.error("❌ Aucun équipement valide trouvé (au moins 2 pannes requises pour la prédiction)")
        return
    
    # Interface de sélection d'équipement
    st.markdown("### Sélection de l'Équipement")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Utiliser le cache pour les codes d'équipements avec un paramètre hashable
        equipment_codes = get_equipment_codes(dashboard.valid_equipments)
        selected_equipment = st.selectbox(
            "Choisir un équipement:",
            equipment_codes,
            help=f"Sélectionnez parmi les {len(equipment_codes)} équipements valides)",
            key="equipment_selector"
        )
    
    with col2:
        prediction_date = st.date_input(
            "Date d'analyse:",
            value=datetime.now().date(),
            help="Date à partir de laquelle faire la prédiction",
            key="prediction_date_input"
        )
    
    # Afficher les informations de l'équipement sélectionné
    if selected_equipment:
        eq_info = dashboard.df_clean[dashboard.df_clean['code_equipement'] == selected_equipment]
        if not eq_info.empty:
            st.markdown("#### Informations de l'Équipement")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Code", selected_equipment)
            with col2:
                eq_type = eq_info['equipement'].iloc[0] if 'equipement' in eq_info.columns else 'N/A'
                st.metric("Type", eq_type)
            with col3:
                total_interventions = len(eq_info)
                st.metric("Total Interventions", total_interventions)
            with col4:
                total_pannes = eq_info['is_panne'].sum() if 'is_panne' in eq_info.columns else 0
                st.metric("Total Pannes", total_pannes)
    
    if st.button("Analyser cet Équipement", type="primary", use_container_width=True, key="analyze_equipment_button"):
        
        # Utiliser le cache intelligent pour les prédictions individuelles
        current_date_str = prediction_date.strftime('%Y-%m-%d')
        model_hash = getattr(st.session_state, 'model_hash', 'no_model')
        data_hash = getattr(st.session_state, 'data_hash', 'no_data')
        
        # Effectuer la prédiction avec cache
        prediction = compute_individual_prediction_cached(
            selected_equipment, 
            current_date_str, 
            model_hash, 
            data_hash
        )
        
        if prediction and 'error' not in prediction:
            
            # Affichage du résultat principal
            st.markdown("### Résultat de la Prédiction")
            
            # Carte de résultat avec style selon le risque
            risk_level_card(
                prediction['risk_level'],
                prediction['days_to_failure'],
                prediction['predicted_failure_date'],
                prediction['recommended_action']
            )
            
            # Détails de l'équipement
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Code Équipement", prediction['equipment_code'])
                st.metric("Type d'Équipement", prediction['equipment_type'])
            
            with col2:
                st.metric("Interventions Totales", prediction['total_interventions'])
                st.metric("Dernière Intervention", prediction['last_intervention'])
            
            with col3:
                confidence = prediction['confidence_interval']
                st.metric("Intervalle de Confiance", f"{confidence['lower']}-{confidence['upper']} jours")
                st.metric("Date d'Analyse", prediction['current_date'])
            
            # Afficher les probabilités si disponibles
            if 'failure_probabilities' in prediction and prediction['failure_probabilities']:
                st.markdown("### Probabilités de Panne")
                probs = prediction['failure_probabilities']
                col1, col2, col3 = st.columns(3)
                
                if '7_days' in probs:
                    with col1:
                        st.metric("Dans 7 jours", f"{probs['7_days']:.1%}")
                if '30_days' in probs:
                    with col2:
                        st.metric("Dans 30 jours", f"{probs['30_days']:.1%}")
                if '90_days' in probs:
                    with col3:
                        st.metric("Dans 90 jours", f"{probs['90_days']:.1%}")
            
            # Afficher les features calculées (pour debug/transparence)
            with st.expander("Détails techniques (Features calculées)", expanded=False):
                features_df = pd.DataFrame([prediction['features']])
                st.dataframe(features_df.T, use_container_width=True)
        
        else:
            st.error(f"Erreur lors de la prédiction: {prediction.get('error', 'Erreur inconnue')}")
