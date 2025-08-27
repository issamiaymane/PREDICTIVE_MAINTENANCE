#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Page de prédiction générale
"""

import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime
from ui.components import page_header, warning_message
from config.settings import ANALYSIS_LIMITS

@st.cache_data
def create_risk_distribution_chart(pred_df_hash):
    """
    Cache pour la création du graphique de distribution des risques
    """
    # Reconstruire le DataFrame à partir du hash
    pred_df = pd.DataFrame(pred_df_hash)
    
    critical_count = len(pred_df[pred_df['days_to_failure'] <= 7])
    high_count = len(pred_df[(pred_df['days_to_failure'] > 7) & (pred_df['days_to_failure'] <= 30)])
    moderate_count = len(pred_df[(pred_df['days_to_failure'] > 30) & (pred_df['days_to_failure'] <= 90)])
    low_count = len(pred_df[pred_df['days_to_failure'] > 90])
    
    risk_data = pd.DataFrame({
        'Niveau': ['Critique', 'Élevé', 'Modéré', 'Faible'],
        'Nombre': [critical_count, high_count, moderate_count, low_count]
    })
    
    fig = px.pie(
        risk_data, 
        values='Nombre', 
        names='Niveau',
        title="Répartition des Niveaux de Risque",
        color_discrete_map={
            'Critique': '#c41e3a',
            'Élevé': '#e67e22',
            'Modéré': '#f1c40f',
            'Faible': '#27ae60'
        }
    )
    fig.update_layout(
        font_color="#1e4a72",
        title_font_size=16,
        title_font_color="#1e4a72"
    )
    return fig

@st.cache_data
def create_priority_timeline_chart(pred_df_hash):
    """
    Cache pour la création du graphique de timeline des priorités
    """
    # Reconstruire le DataFrame à partir du hash
    pred_df = pd.DataFrame(pred_df_hash)
    top_priority = pred_df.nsmallest(15, 'days_to_failure')
    
    fig = px.bar(
        top_priority,
        x='days_to_failure',
        y='equipment_code',
        orientation='h',
        title="Top 15 - Équipements Prioritaires",
        labels={'days_to_failure': 'Jours avant panne', 'equipment_code': 'Équipement'},
        color='days_to_failure',
        color_continuous_scale=['#c41e3a', '#e67e22', '#f1c40f', '#27ae60']
    )
    fig.update_layout(
        yaxis={'categoryorder': 'total ascending'},
        font_color="#1e4a72",
        title_font_size=16,
        title_font_color="#1e4a72"
    )
    return fig

@st.cache_data
def create_failure_delay_histogram(pred_df_hash):
    """
    Cache pour la création de l'histogramme des délais
    """
    # Reconstruire le DataFrame à partir du hash
    pred_df = pd.DataFrame(pred_df_hash)
    
    fig = px.histogram(
        pred_df,
        x='days_to_failure',
        nbins=30,
        title="Distribution des Délais Avant Panne",
        labels={'days_to_failure': 'Jours avant panne', 'count': 'Nombre d\'équipements'},
        color_discrete_sequence=['#1e4a72']
    )
    fig.update_layout(
        font_color="#1e4a72",
        title_font_size=16,
        title_font_color="#1e4a72"
    )
    return fig

def prediction_page():
    """
    Page de prédiction générale avec cache intelligent
    """
    page_header("Prédiction Générale", "Vue d'ensemble des prédictions de pannes pour les équipements valides")
    
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
    
    # Configuration de l'analyse
    st.markdown("### Configuration de l'Analyse")
    col1, col2, col3 = st.columns([2, 1, 1])
    
    max_equipments = len(dashboard.valid_equipments)
    
    with col1:
        num_equipments = st.slider(
            "Nombre d'équipements à analyser",
            min_value=10,
            max_value=min(ANALYSIS_LIMITS['max_equipments'], max_equipments),
            value=min(100, max_equipments),
            step=10,
            help=f"Maximum disponible: {max_equipments} équipements valides",
            key="num_equipments_slider"
        )
    
    with col2:
        analysis_date = st.date_input(
            "Date d'analyse:",
            value=datetime.now().date(),
            help="Date à partir de laquelle faire la prédiction",
            key="analysis_date_input"
        )
    
    with col3:
        st.write("")
        st.write("")
        analyze_button = st.button("Lancer l'Analyse", type="primary", use_container_width=True, key="analyze_button")
    
    # Afficher les statistiques des équipements valides
    st.markdown("### Statistiques des Équipements")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Équipements", f"{dashboard.df_clean['code_equipement'].nunique():,}")
    with col2:
        st.metric("Équipements Valides", f"{len(dashboard.valid_equipments):,}", help="≥2 pannes historiques")
    with col3:
        valid_ratio = len(dashboard.valid_equipments) / dashboard.df_clean['code_equipement'].nunique() * 100
        st.metric("Taux de Validité", f"{valid_ratio:.1f}%")
    with col4:
        st.metric("Observations ML", f"{len(dashboard.model_df):,}", help="Données temporelles")
    
    if analyze_button:
        equipment_sample = dashboard.valid_equipments[:num_equipments]
        current_date_str = analysis_date.strftime('%Y-%m-%d')
        
        # Utiliser le cache intelligent pour les prédictions
        model_hash = getattr(st.session_state, 'model_hash', 'no_model')
        data_hash = getattr(st.session_state, 'data_hash', 'no_data')
        
        st.markdown("### Analyse en Cours...")
        
        # Utiliser la fonction de cache depuis app.py
        from app import compute_predictions_cached
        predictions_list = compute_predictions_cached(
            equipment_sample, 
            current_date_str, 
            model_hash, 
            data_hash
        )
        
        if predictions_list and len(predictions_list) > 0:
            pred_df = pd.DataFrame(predictions_list)
            
            # Statistiques globales
            st.markdown("### Vue d'Ensemble des Risques")
            
            critical_count = len(pred_df[pred_df['days_to_failure'] <= 7])
            high_count = len(pred_df[(pred_df['days_to_failure'] > 7) & (pred_df['days_to_failure'] <= 30)])
            moderate_count = len(pred_df[(pred_df['days_to_failure'] > 30) & (pred_df['days_to_failure'] <= 90)])
            low_count = len(pred_df[pred_df['days_to_failure'] > 90])
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Critique", critical_count, help="≤ 7 jours", delta=f"{critical_count/len(pred_df)*100:.1f}%")
            with col2:
                st.metric("Élevé", high_count, help="8-30 jours", delta=f"{high_count/len(pred_df)*100:.1f}%")
            with col3:
                st.metric("Modéré", moderate_count, help="31-90 jours", delta=f"{moderate_count/len(pred_df)*100:.1f}%")
            with col4:
                st.metric("Faible", low_count, help="> 90 jours", delta=f"{low_count/len(pred_df)*100:.1f}%")
            
            # Visualisations avec cache intelligent
            st.markdown("### Visualisations des Prédictions")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Distribution des risques (Pie Chart) - avec cache
                pred_df_hash = pred_df.to_dict('records')  # Convertir en format hashable
                fig_pie = create_risk_distribution_chart(pred_df_hash)
                st.plotly_chart(fig_pie, use_container_width=True)
            
            with col2:
                # Timeline des maintenances prioritaires - avec cache
                fig_bar = create_priority_timeline_chart(pred_df_hash)
                st.plotly_chart(fig_bar, use_container_width=True)
            
            # Distribution histogramme - avec cache
            st.markdown("### Distribution des Délais de Panne")
            fig_hist = create_failure_delay_histogram(pred_df_hash)
            st.plotly_chart(fig_hist, use_container_width=True)
            
            # Analyse par type d'équipement
            if 'equipment_type' in pred_df.columns:
                st.markdown("### Analyse par Type d'Équipement")
                
                type_analysis = pred_df.groupby('equipment_type').agg({
                    'days_to_failure': ['count', 'mean', 'min'],
                    'risk_class': lambda x: (x == 'critical').sum()
                }).round(2)
                
                type_analysis.columns = ['Nombre', 'Délai Moyen', 'Délai Min', 'Critiques']
                type_analysis = type_analysis.reset_index()
                
                fig_scatter = px.scatter(
                    type_analysis,
                    x='Délai Moyen',
                    y='Nombre',
                    size='Critiques',
                    hover_name='equipment_type',
                    title="Performance par Type d'Équipement",
                    labels={
                        'Délai Moyen': 'Délai moyen avant panne (jours)',
                        'Nombre': 'Nombre d\'équipements'
                    },
                    color='Critiques',
                    color_continuous_scale=['#27ae60', '#e67e22', '#c41e3a']
                )
                fig_scatter.update_layout(
                    font_color="#1e4a72",
                    title_font_size=16,
                    title_font_color="#1e4a72"
                )
                st.plotly_chart(fig_scatter, use_container_width=True)
            
            # Tableau détaillé
            st.markdown("### Tableau Complet des Prédictions")
            
            # Préparer les données pour l'affichage
            display_df = pred_df[[
                'equipment_code', 'equipment_type', 'days_to_failure', 
                'predicted_failure_date', 'risk_level', 'recommended_action'
            ]].copy()
            
            display_df.columns = [
                'Code Équipement', 
                'Type', 
                'Jours avant panne', 
                'Date prévue', 
                'Niveau de risque', 
                'Action recommandée'
            ]
            
            # Trier par priorité (moins de jours d'abord)
            display_df = display_df.sort_values('Jours avant panne').reset_index(drop=True)
            
            st.markdown(f"**Total:** {len(display_df)} équipements analysés (sur {len(dashboard.valid_equipments)} valides)")
            
            # Afficher le tableau complet
            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True
            )
            
            # Export des résultats
            st.markdown("### Export des Résultats")
            col1, col2 = st.columns(2)
            
            with col1:
                csv = pred_df.to_csv(index=False)
                st.download_button(
                    label="📥 Télécharger CSV Complet",
                    data=csv,
                    file_name=f"predictions_dashboard_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    key="download_full_csv"
                )
            
            with col2:
                # Export des équipements critiques seulement
                critical_df = pred_df[pred_df['risk_level'] == 'CRITIQUE']
                if len(critical_df) > 0:
                    critical_csv = critical_df.to_csv(index=False)
                    st.download_button(
                        label="🚨 Télécharger Critiques Seulement",
                        data=critical_csv,
                        file_name=f"critiques_snrt_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        key="download_critical_csv"
                    )
                
        else:
            st.error("Impossible de générer des prédictions. Vérifiez les données et le modèle.")
