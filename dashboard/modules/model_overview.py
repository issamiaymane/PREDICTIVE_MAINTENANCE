#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Page d'aperçu du modèle
"""

import streamlit as st
import pandas as pd
from ui.components import page_header, warning_message
from auth.authorization import require_admin

def model_overview_page():
    """
    Page d'aperçu du modèle
    """
    # Vérifier les permissions d'administrateur
    require_admin()
    
    page_header("Aperçu du Modèle", "Informations détaillées sur le modèle de machine learning")
    
    # Vérifier si le modèle est chargé
    dashboard = st.session_state.dashboard
    model_loaded = dashboard.model_package is not None
    
    if not model_loaded:
        warning_message("Veuillez charger le modèle et les données dans la barre latérale.")
        return
    
    model_package = dashboard.model_package
    
    # Informations générales du modèle
    st.markdown("### Informations Générales")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        model_name = model_package.get('model_name', 'Unknown')
        st.metric("Type de Modèle", model_name)
    
    with col2:
        feature_count = len(model_package.get('feature_columns', []))
        st.metric("Nombre de Features", feature_count)
    
    with col3:
        training_info = model_package.get('training_info', {})
        training_date = training_info.get('date', 'N/A')
        if training_date != 'N/A':
            try:
                formatted_date = pd.to_datetime(training_date).strftime('%d/%m/%Y')
                st.metric("Date d'Entraînement", formatted_date)
            except:
                st.metric("Date d'Entraînement", training_date)
        else:
            st.metric("Date d'Entraînement", training_date)
    
    with col4:
        n_samples_train = training_info.get('n_samples_train', 'N/A')
        st.metric("Échantillons d'Entraînement", f"{n_samples_train:,}" if isinstance(n_samples_train, (int, float)) else n_samples_train)
    
    # Features du modèle
    st.markdown("### Features du Modèle")
    features = model_package.get('feature_columns', [])
    if features:
        col1, col2 = st.columns(2)
        half = len(features) // 2
        with col1:
            for f in features[:half]:
                st.write(f"• {f}")
        with col2:
            for f in features[half:]:
                st.write(f"• {f}")
    
    # Métriques de performance
    st.markdown("### Performance du Modèle")
    metrics = model_package.get('performance_metrics', {})
    if metrics:
        col1, col2, col3 = st.columns(3)
        with col1:
            mae = metrics.get('mae', 'N/A')
            if mae != 'N/A':
                st.metric("MAE (Mean Absolute Error)", f"{mae:.1f} jours")
            else:
                st.metric("MAE", mae)
        with col2:
            rmse = metrics.get('rmse', 'N/A')
            if rmse != 'N/A':
                st.metric("RMSE (Root Mean Square Error)", f"{rmse:.1f} jours")
            else:
                st.metric("RMSE", rmse)
        with col3:
            r2 = metrics.get('r2', 'N/A')
            if r2 != 'N/A':
                st.metric("R² Score", f"{r2:.3f}")
            else:
                st.metric("R² Score", r2)
    
    # Critère de sélection des équipements
    st.markdown("### Critères de Sélection")
    st.info("**Critère de validité:** Seuls les équipements avec au moins 2 pannes historiques sont inclus dans l'analyse prédictive.")
    
    # Vérifier si les données sont chargées pour afficher les statistiques
    data_loaded = dashboard.df_clean is not None
    if data_loaded:
        if dashboard.valid_equipments is not None:
            total_eq = dashboard.df_clean['code_equipement'].nunique()
            valid_eq = len(dashboard.valid_equipments)
            st.write(f"Équipements totaux: **{total_eq:,}**")
            st.write(f"Équipements valides: **{valid_eq:,}** ({valid_eq/total_eq*100:.1f}%)")
            st.write(f"Équipements exclus: **{total_eq - valid_eq:,}** (moins de 2 pannes)")
    
    # Informations sur le modèle de survie si disponible
    if 'survival_model' in model_package:
        st.markdown("### Modèle de Survie")
        survival_info = model_package['survival_model']
        median_survival = survival_info.get('median_survival_time', 'N/A')
        if median_survival != 'N/A':
            st.info(f"Temps médian de survie (sans panne): {median_survival:.0f} jours")
