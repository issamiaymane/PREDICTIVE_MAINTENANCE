#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Composants réutilisables pour l'interface utilisateur
"""

import streamlit as st
from config.settings import SNRT_COLORS

def snrt_header():
    """
    Affiche l'en-tête SNRT avec les couleurs de la marque
    """
    st.markdown("""
    <div class="main-header">
        <div class="snrt-colors">
            <div class="color-dot color-blue"></div>
            <div class="color-dot color-red"></div>
            <div class="color-dot color-orange"></div>
            <div class="color-dot color-green"></div>
        </div>
        <h1>Système de Maintenance Prédictive Intelligente</h1>
        <p>Société Nationale de Radiodiffusion et de Télévision</p>
    </div>
    """, unsafe_allow_html=True)

def snrt_colors_display():
    """
    Affiche les couleurs SNRT
    """
    st.markdown("""
    <div class="snrt-colors">
        <div class="color-dot color-blue"></div>
        <div class="color-dot color-red"></div>
        <div class="color-dot color-orange"></div>
        <div class="color-dot color-green"></div>
    </div>
    """, unsafe_allow_html=True)

def page_header(title, description):
    """
    Affiche un en-tête de page
    Args:
        title (str): Titre de la page
        description (str): Description de la page
    """
    st.markdown(f"""
    <div class="page-header">
        <h2>{title}</h2>
        <p>{description}</p>
    </div>
    """, unsafe_allow_html=True)

def risk_level_card(risk_level, days_to_failure, predicted_date, action):
    """
    Affiche une carte de niveau de risque
    Args:
        risk_level (str): Niveau de risque
        days_to_failure (int): Jours avant panne
        predicted_date (str): Date prédite
        action (str): Action recommandée
    """
    risk_class = risk_level.lower()
    st.markdown(f"""
    <div class="risk-{risk_class}">
        <h3>Niveau de Risque: {risk_level}</h3>
        <h2>{days_to_failure} jours avant panne prévue</h2>
        <p>Date prévue: {predicted_date}</p>
        <p><strong>Action recommandée:</strong> {action}</p>
    </div>
    """, unsafe_allow_html=True)

def metric_card(title, value, delta=None, help_text=None):
    """
    Affiche une carte de métrique
    Args:
        title (str): Titre de la métrique
        value: Valeur de la métrique
        delta: Variation (optionnel)
        help_text (str): Texte d'aide (optionnel)
    """
    st.metric(title, value, delta=delta, help=help_text)

def info_card(title, content):
    """
    Affiche une carte d'information
    Args:
        title (str): Titre de la carte
        content (str): Contenu de la carte
    """
    st.markdown(f"""
    <div class="info-card">
        <h4>{title}</h4>
        <p>{content}</p>
    </div>
    """, unsafe_allow_html=True)

def success_message(message):
    """
    Affiche un message de succès
    Args:
        message (str): Message à afficher
    """
    st.markdown(f"""
    <div class="success-card">
        ✅ {message}
    </div>
    """, unsafe_allow_html=True)

def warning_message(message):
    """
    Affiche un message d'avertissement
    Args:
        message (str): Message à afficher
    """
    st.markdown(f"""
    <div class="warning-card">
        ⚠️ {message}
    </div>
    """, unsafe_allow_html=True)

def user_info_display(username, name, role):
    """
    Affiche les informations de l'utilisateur
    Args:
        username (str): Nom d'utilisateur
        name (str): Nom complet
        role (str): Rôle de l'utilisateur
    """
    st.markdown("### Informations de l'Utilisateur")
    st.write(f"Nom d'utilisateur : **@{username}**")
    st.write(f"Nom et Prénom : **{name}**")
    st.write(f"Role : **{role.title()}**")

def model_info_display(model_package):
    """
    Affiche les informations du modèle
    Args:
        model_package (dict): Package du modèle
    """
    if model_package:
        model_name = model_package.get('model_name', 'Unknown')
        st.info(f"Modèle: **{model_name}**")
        
        metrics = model_package.get('performance_metrics', {})
        if metrics:
            col1, col2 = st.columns(2)
            with col1:
                mae = metrics.get('mae', 'N/A')
                if mae != 'N/A':
                    st.metric("MAE", f"{mae:.1f}j")
                else:
                    st.metric("MAE", mae)
            with col2:
                r2 = metrics.get('r2', 'N/A')
                if r2 != 'N/A':
                    st.metric("R²", f"{r2:.3f}")
                else:
                    st.metric("R²", r2)

def data_info_display(dashboard):
    """
    Affiche les informations des données
    Args:
        dashboard: Instance de PredictionDashboard
    """
    if (hasattr(dashboard, 'model_df') and dashboard.model_df is not None and
        hasattr(dashboard, 'valid_equipments') and dashboard.valid_equipments is not None):
        st.info(f"Données : ***{len(dashboard.model_df):,} Données temporelles dont {len(dashboard.valid_equipments)} Équipements valides***")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Lignes", f"{len(dashboard.df_clean):,}")
    with col2:
        st.metric("Équipements", f"{dashboard.df_clean['code_equipement'].nunique():,}")

def snrt_footer():
    """
    Affiche le pied de page SNRT
    """
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; padding: 2rem; background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%); border-radius: 15px; margin-top: 2rem;">
        <div class="snrt-colors" style="justify-content: center; margin-bottom: 1rem;">
            <div class="color-dot color-blue"></div>
            <div class="color-dot color-red"></div>
            <div class="color-dot color-orange"></div>
            <div class="color-dot color-green"></div>
        </div>
        <h4 style="color: #1e4a72; margin-bottom: 0.5rem;">SNRT - DISI</h4>
        <p style="color: #7f8c8d; margin: 0;">Système de Maintenance Prédictive Intelligente</p>
        <small style="color: #95a5a6;">Version : V1.0</small>
    </div>
    """, unsafe_allow_html=True)
