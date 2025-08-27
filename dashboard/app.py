#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Système de Maintenance Prédictive Intelligente
Auteur: Aymane ISSAMI
Organisation: Société Nationale de Radiodiffusion et de Télévision du Maroc
Description: Système de Maintenance Prédictive Intelligente visant à prédire le moment probable de défaillance d'un équipement.
Version: 1.0 - Version Restructurée avec Cache Intelligent
"""

import streamlit as st
import warnings
warnings.filterwarnings('ignore')

# Configuration
from config.settings import setup_page_config
from config.database import init_mongodb

# Authentification et autorisation
from auth.authentication import is_authenticated, get_current_user, logout_user
from auth.authorization import can_access_advanced_features

# Modèles
from models.prediction_model import PredictionDashboard

# Interface utilisateur
from ui.styles import apply_styles
from ui.components import snrt_header, snrt_footer, user_info_display, model_info_display, data_info_display

# Pages
from modules.login import login_page
from modules.prediction_general import prediction_page
from modules.prediction_individual import single_equipment_prediction_page
from modules.data_analysis import data_analysis_page
from modules.model_overview import model_overview_page

@st.cache_data
def load_model_cached(model_file_bytes, model_file_name):
    """
    Cache intelligent pour le chargement du modèle
    """
    import io
    import hashlib
    
    # Créer un hash unique du fichier pour le cache
    file_hash = hashlib.md5(model_file_bytes).hexdigest()
    
    dashboard = PredictionDashboard()
    model_file = io.BytesIO(model_file_bytes)
    model_file.name = model_file_name
    
    if dashboard.load_model(model_file):
        return {
            'model_package': dashboard.model_package,
            'file_hash': file_hash
        }
    return None

@st.cache_data
def load_data_cached(data_file_bytes, data_file_name):
    """
    Cache intelligent pour le chargement des données
    """
    import io
    import hashlib
    
    # Créer un hash unique du fichier pour le cache
    file_hash = hashlib.md5(data_file_bytes).hexdigest()
    
    dashboard = PredictionDashboard()
    data_file = io.BytesIO(data_file_bytes)
    data_file.name = data_file_name
    
    if dashboard.load_data(data_file):
        return {
            'df_clean': dashboard.df_clean,
            'model_df': dashboard.model_df,
            'valid_equipments': dashboard.valid_equipments,
            'file_hash': file_hash
        }
    return None

@st.cache_data
def compute_predictions_cached(equipment_codes, current_date_str, model_hash, data_hash):
    """
    Cache intelligent pour les prédictions - calculé une seule fois par combinaison de paramètres
    """
    dashboard = st.session_state.dashboard
    
    # Vérifier si on a déjà les prédictions en cache
    cache_key = f"predictions_{model_hash}_{data_hash}_{current_date_str}_{len(equipment_codes)}"
    
    if 'predictions_cache' not in st.session_state:
        st.session_state.predictions_cache = {}
    
    if cache_key in st.session_state.predictions_cache:
        return st.session_state.predictions_cache[cache_key]
    
    # Calculer les prédictions
    predictions_list = dashboard.safe_predict_batch(equipment_codes)
    
    # Stocker en cache
    st.session_state.predictions_cache[cache_key] = predictions_list
    
    return predictions_list

def create_simple_sidebar():
    """
    Crée une sidebar simple sans navigation, juste avec les uploaders et infos utilisateur
    """
    with st.sidebar:
        
        st.markdown("""
        <div style="text-align: center; margin-bottom: 2rem;">
            <div class="snrt-colors">
                <div class="color-dot color-blue"></div>
                <div class="color-dot color-red"></div>
                <div class="color-dot color-orange"></div>
                <div class="color-dot color-green"></div>
            </div>
            <h3 style="color: #1e4a72; margin: 0;">Système de Maintenance Prédictive Intelligente</h3>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        # Informations de l'utilisateur
        current_user = get_current_user()
        if current_user:
            user_info_display(
                current_user['username'],
                current_user['name'],
                current_user['role']
            )

        st.markdown("---")
        
        # Chargement du modèle avec cache intelligent
        st.markdown("### Modèle de Prédiction")
        model_file = st.file_uploader("Charger le modèle (.pkl)", type=['pkl'], key="model_uploader")
        model_loaded = False
        
        if model_file is not None:
            # Lire le fichier en bytes pour le cache
            model_file.seek(0)
            model_bytes = model_file.read()
            
            # Utiliser le cache intelligent
            model_result = load_model_cached(model_bytes, model_file.name)
            if model_result:
                st.success("Modèle chargé avec succès!")
                model_loaded = True
                st.session_state.dashboard.model_package = model_result['model_package']
                st.session_state.model_hash = model_result['file_hash']
                
                model_info_display(model_result['model_package'])
        
        st.markdown("---")
        
        # Chargement des données avec cache intelligent
        st.markdown("### Données Historiques")
        data_file = st.file_uploader("Charger les données", type=['csv', 'xlsx'], key="data_uploader")
        data_loaded = False
        
        if data_file is not None:
            # Lire le fichier en bytes pour le cache
            data_file.seek(0)
            data_bytes = data_file.read()
            
            # Utiliser le cache intelligent
            data_result = load_data_cached(data_bytes, data_file.name)
            if data_result:
                st.success("Données chargées avec succès!")
                data_loaded = True
                
                # Mettre à jour le dashboard avec les données en cache
                st.session_state.dashboard.df_clean = data_result['df_clean']
                st.session_state.dashboard.model_df = data_result['model_df']
                st.session_state.dashboard.valid_equipments = data_result['valid_equipments']
                st.session_state.data_hash = data_result['file_hash']
                
                data_info_display(st.session_state.dashboard)

        st.markdown("---")
        
        # Bouton de déconnexion
        if st.button("Se déconnecter", key="logout_button"):
            logout_user()
            st.rerun()
        
        return model_loaded, data_loaded

def main_dashboard():
    """
    Dashboard principal après connexion - avec navigation par onglets pour une navigation instantanée
    """
    # En-tête principal
    snrt_header()
    
    # Initialiser le dashboard une seule fois
    if 'dashboard' not in st.session_state:
        st.session_state.dashboard = PredictionDashboard()
    
    # Créer la sidebar simple
    model_loaded, data_loaded = create_simple_sidebar()
    
    # Navigation par onglets pour une navigation instantanée
    st.markdown("### Navigation")
    
    # Créer les onglets avec les pages disponibles
    tab_names = ["Prédictions Générales", "Prédiction Individuelle"]
    
    # Ajouter les onglets avancés seulement pour les administrateurs
    if can_access_advanced_features():
        tab_names.extend(["Analyse des Données", "Aperçu du Modèle"])
    else:
        st.info("ℹ️ Accès limité: Les fonctionnalités d'analyse avancée sont réservées aux administrateurs.")
    
    # Créer les onglets
    tabs = st.tabs(tab_names)
    
    # Onglet 1: Prédictions Générales
    with tabs[0]:
        prediction_page()
    
    # Onglet 2: Prédiction Individuelle
    with tabs[1]:
        single_equipment_prediction_page()
    
    # Onglets 3 et 4: Fonctionnalités avancées (si autorisé)
    if can_access_advanced_features():
        with tabs[2]:
            data_analysis_page()
        
        with tabs[3]:
            model_overview_page()
    
    # Footer SNRT
    snrt_footer()

def main():
    """
    Fonction principale avec gestion de l'authentification
    """
    # Configuration de la page
    setup_page_config()
    
    # Appliquer les styles CSS
    apply_styles()
    
    # Initialiser la connexion MongoDB
    users_collection, db_connected = init_mongodb()
    
    # Initialiser l'état de session
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    
    # Afficher la page appropriée
    if st.session_state.authenticated:
        main_dashboard()
    else:
        login_page(users_collection, db_connected)

if __name__ == "__main__":
    main()
