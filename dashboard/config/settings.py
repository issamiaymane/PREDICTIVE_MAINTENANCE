#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Configuration générale du système
"""

import streamlit as st

# Configuration de la page Streamlit
PAGE_CONFIG = {
    "page_title": "Système de Maintenance Prédictive Intelligente",
    "page_icon": "📺",
    "layout": "wide",
    "initial_sidebar_state": "expanded"
}

# Configuration MongoDB
MONGODB_CONFIG = {
    "host": "localhost",
    "port": 27017,
    "database": "snrt_app",
    "collection": "users",
    "timeout": 5000
}

# Configuration des rôles
ROLES = {
    "admin": "Administrateur",
    "user": "Utilisateur"
}

# Configuration des niveaux de risque
RISK_LEVELS = {
    "critical": {
        "name": "CRITIQUE",
        "max_days": 7,
        "action": "Maintenance d'urgence requise",
        "color": "#c41e3a"
    },
    "high": {
        "name": "ÉLEVÉ",
        "max_days": 30,
        "action": "Planifier maintenance dans les 2 semaines",
        "color": "#e67e22"
    },
    "moderate": {
        "name": "MODÉRÉ",
        "max_days": 90,
        "action": "Surveillance accrue recommandée",
        "color": "#f1c40f"
    },
    "low": {
        "name": "FAIBLE",
        "max_days": float('inf'),
        "action": "Maintenance préventive standard",
        "color": "#27ae60"
    }
}

# Configuration des couleurs SNRT
SNRT_COLORS = {
    "blue": "#1e4a72",
    "red": "#c41e3a",
    "orange": "#e67e22",
    "green": "#27ae60"
}

# Configuration des probabilités par défaut
DEFAULT_PROBABILITIES = {
    '7_days': 0.778,   # 77.8%
    '30_days': 0.902,  # 90.2%
    '90_days': 0.961   # 96.1%
}

# Configuration des intervalles de confiance
CONFIDENCE_INTERVAL_FACTOR = {
    'lower': 0.7,
    'upper': 1.3
}

# Configuration des limites d'analyse
ANALYSIS_LIMITS = {
    'max_equipments': 500,
    'max_errors': 10,
    'min_failures_for_prediction': 2
}

def setup_page_config():
    """Configure la page Streamlit"""
    st.set_page_config(**PAGE_CONFIG)
