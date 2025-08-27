#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Page d'analyse des données
"""

import streamlit as st
import pandas as pd
import plotly.express as px
from ui.components import page_header, warning_message
from auth.authorization import require_admin

def data_analysis_page():
    """
    Page d'analyse des données
    """
    # Vérifier les permissions d'administrateur
    require_admin()
    
    page_header("Analyse des Données", "Exploration et analyse statistique du dataset historique")
    
    # Vérifier si les données sont chargées
    dashboard = st.session_state.dashboard
    data_loaded = dashboard.df_clean is not None
    
    if not data_loaded:
        warning_message("Veuillez charger le modèle et les données dans la barre latérale.")
        return
    
    df = dashboard.df_clean
    
    # Vue d'ensemble des données
    st.markdown("### Vue d'Ensemble du Dataset")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Enregistrements", f"{len(df):,}")
    with col2:
        st.metric("Équipements Uniques", f"{df['code_equipement'].nunique():,}")
    with col3:
        if 'DateDebut' in df.columns:
            period_days = (df['DateDebut'].max() - df['DateDebut'].min()).days
            st.metric("Période (jours)", f"{period_days:,}")
        else:
            st.metric("Période", "N/A")
    with col4:
        avg_interventions = len(df) / df['code_equipement'].nunique() if df['code_equipement'].nunique() > 0 else 0
        st.metric("Moy. Interventions/Équip.", f"{avg_interventions:.1f}")
    
    # Informations sur la période
    if 'DateDebut' in df.columns:
        start_date = df['DateDebut'].min().strftime('%d/%m/%Y')
        end_date = df['DateDebut'].max().strftime('%d/%m/%Y')
        st.info(f"**Période couverte:** {start_date} → {end_date}")
    
    # Analyse des équipements valides
    st.markdown("### Analyse de Validité des Équipements")
    
    if dashboard.valid_equipments is not None:
        total_equipments = df['code_equipement'].nunique()
        valid_equipments = len(dashboard.valid_equipments)
        invalid_equipments = total_equipments - valid_equipments
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Équipements Valides", f"{valid_equipments:,}", help="≥2 pannes")
        with col2:
            st.metric("Équipements Exclus", f"{invalid_equipments:,}", help="<2 pannes")
        with col3:
            valid_ratio = (valid_equipments / total_equipments) * 100
            st.metric("Taux de Validité", f"{valid_ratio:.1f}%")
        
        # Graphique de répartition
        fig_validity = px.pie(
            values=[valid_equipments, invalid_equipments],
            names=['Valides (≥2 pannes)', 'Exclus (<2 pannes)'],
            title="Répartition des Équipements - Critère de Validité",
            color_discrete_map={'Valides (≥2 pannes)': '#27ae60', 'Exclus (<2 pannes)': '#e74c3c'}
        )
        st.plotly_chart(fig_validity, use_container_width=True)
    
    # Statistiques sur les pannes
    if 'is_panne' in df.columns:
        st.markdown("### Statistiques des Pannes")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Pannes", f"{df['is_panne'].sum():,}")
        with col2:
            st.metric("Taux de Panne", f"{df['is_panne'].mean()*100:.1f}%")
        with col3:
            if hasattr(dashboard, 'model_df') and dashboard.model_df is not None:
                median_time = dashboard.model_df['days_to_next_failure'].median()
                st.metric("Temps Médian entre Pannes", f"{median_time:.0f} jours")
    
    # Analyse du dataset enrichi (model_df)
    if hasattr(dashboard, 'model_df') and dashboard.model_df is not None:
        st.markdown("### Dataset Enrichi (Features Temporelles)")
        model_df = dashboard.model_df
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Observations Temporelles", f"{len(model_df):,}")
        with col2:
            st.metric("Features Créées", f"{len([col for col in model_df.columns if col not in ['code_equipement', 'equipement_type', 'date_actuelle']])}")
        with col3:
            st.metric("Équipements dans le Modèle", f"{model_df['code_equipement'].nunique():,}")
        
        # Distribution des délais entre pannes
        fig_delays = px.histogram(
            model_df,
            x='days_to_next_failure',
            title="Distribution des Délais entre Pannes Consécutives",
            labels={'days_to_next_failure': 'Jours jusqu\'à la prochaine panne', 'count': 'Fréquence'},
            color_discrete_sequence=['#1e4a72']
        )
        fig_delays.update_layout(
            font_color="#1e4a72",
            title_font_size=16,
            title_font_color="#1e4a72"
        )
        st.plotly_chart(fig_delays, use_container_width=True)
        
        # Afficher les statistiques détaillées
        st.markdown("**Statistiques des délais entre pannes:**")
        stats = model_df['days_to_next_failure'].describe()
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Moyenne", f"{stats['mean']:.1f} jours")
        with col2:
            st.metric("Médiane", f"{stats['50%']:.1f} jours")
        with col3:
            st.metric("Min", f"{stats['min']:.0f} jours")
        with col4:
            st.metric("Max", f"{stats['max']:.0f} jours")
    
    # Structure du dataset
    st.markdown("### Structure du Dataset")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Informations sur les colonnes:**")
        info_df = pd.DataFrame({
            'Colonne': df.columns,
            'Type': df.dtypes.astype(str),
            'Non-Null': df.count(),
            'Null (%)': ((len(df) - df.count()) / len(df) * 100).round(2)
        })
        st.dataframe(info_df, use_container_width=True)
    
    with col2:
        st.markdown("**Aperçu des données:**")
        st.dataframe(df.head(10), use_container_width=True)
