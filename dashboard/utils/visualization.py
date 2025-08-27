#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fonctions de visualisation
"""

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from typing import Dict, List, Optional

def create_risk_distribution_chart(pred_df: pd.DataFrame) -> go.Figure:
    """
    Crée un graphique de distribution des risques
    Args:
        pred_df (pd.DataFrame): DataFrame des prédictions
    Returns:
        go.Figure: Graphique Plotly
    """
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

def create_priority_timeline_chart(pred_df: pd.DataFrame, top_n: int = 15) -> go.Figure:
    """
    Crée un graphique de timeline des équipements prioritaires
    Args:
        pred_df (pd.DataFrame): DataFrame des prédictions
        top_n (int): Nombre d'équipements à afficher
    Returns:
        go.Figure: Graphique Plotly
    """
    top_priority = pred_df.nsmallest(top_n, 'days_to_failure')
    
    fig = px.bar(
        top_priority,
        x='days_to_failure',
        y='equipment_code',
        orientation='h',
        title=f"Top {top_n} - Équipements Prioritaires",
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

def create_failure_delay_histogram(pred_df: pd.DataFrame, nbins: int = 30) -> go.Figure:
    """
    Crée un histogramme de distribution des délais de panne
    Args:
        pred_df (pd.DataFrame): DataFrame des prédictions
        nbins (int): Nombre de bins pour l'histogramme
    Returns:
        go.Figure: Graphique Plotly
    """
    fig = px.histogram(
        pred_df,
        x='days_to_failure',
        nbins=nbins,
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

def create_equipment_type_analysis(pred_df: pd.DataFrame) -> go.Figure:
    """
    Crée un graphique d'analyse par type d'équipement
    Args:
        pred_df (pd.DataFrame): DataFrame des prédictions
    Returns:
        go.Figure: Graphique Plotly
    """
    if 'equipment_type' not in pred_df.columns:
        return None
    
    type_analysis = pred_df.groupby('equipment_type').agg({
        'days_to_failure': ['count', 'mean', 'min'],
        'risk_class': lambda x: (x == 'critical').sum()
    }).round(2)
    
    type_analysis.columns = ['Nombre', 'Délai Moyen', 'Délai Min', 'Critiques']
    type_analysis = type_analysis.reset_index()
    
    fig = px.scatter(
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
    
    fig.update_layout(
        font_color="#1e4a72",
        title_font_size=16,
        title_font_color="#1e4a72"
    )
    
    return fig

def create_validity_pie_chart(valid_equipments: int, invalid_equipments: int) -> go.Figure:
    """
    Crée un graphique en camembert pour la répartition des équipements valides
    Args:
        valid_equipments (int): Nombre d'équipements valides
        invalid_equipments (int): Nombre d'équipements invalides
    Returns:
        go.Figure: Graphique Plotly
    """
    fig = px.pie(
        values=[valid_equipments, invalid_equipments],
        names=['Valides (≥2 pannes)', 'Exclus (<2 pannes)'],
        title="Répartition des Équipements - Critère de Validité",
        color_discrete_map={'Valides (≥2 pannes)': '#27ae60', 'Exclus (<2 pannes)': '#e74c3c'}
    )
    
    fig.update_layout(
        font_color="#1e4a72",
        title_font_size=16,
        title_font_color="#1e4a72"
    )
    
    return fig

def create_delay_distribution_chart(model_df: pd.DataFrame) -> go.Figure:
    """
    Crée un graphique de distribution des délais entre pannes
    Args:
        model_df (pd.DataFrame): DataFrame du modèle enrichi
    Returns:
        go.Figure: Graphique Plotly
    """
    fig = px.histogram(
        model_df,
        x='days_to_next_failure',
        title="Distribution des Délais entre Pannes Consécutives",
        labels={'days_to_next_failure': 'Jours jusqu\'à la prochaine panne', 'count': 'Fréquence'},
        color_discrete_sequence=['#1e4a72']
    )
    
    fig.update_layout(
        font_color="#1e4a72",
        title_font_size=16,
        title_font_color="#1e4a72"
    )
    
    return fig

def create_dashboard_summary_charts(pred_df: pd.DataFrame) -> Dict[str, go.Figure]:
    """
    Crée un ensemble de graphiques pour le résumé du dashboard
    Args:
        pred_df (pd.DataFrame): DataFrame des prédictions
    Returns:
        Dict[str, go.Figure]: Dictionnaire des graphiques
    """
    charts = {}
    
    # Distribution des risques
    charts['risk_distribution'] = create_risk_distribution_chart(pred_df)
    
    # Timeline des priorités
    charts['priority_timeline'] = create_priority_timeline_chart(pred_df)
    
    # Histogramme des délais
    charts['delay_histogram'] = create_failure_delay_histogram(pred_df)
    
    # Analyse par type d'équipement
    if 'equipment_type' in pred_df.columns:
        charts['equipment_type_analysis'] = create_equipment_type_analysis(pred_df)
    
    return charts
