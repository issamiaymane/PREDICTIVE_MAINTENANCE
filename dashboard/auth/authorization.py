#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gestion des rôles et permissions
"""

import streamlit as st
from config.settings import ROLES

def has_role(required_role):
    """
    Vérifie si l'utilisateur a le rôle requis
    Args:
        required_role (str): Rôle requis
    Returns:
        bool: True si l'utilisateur a le rôle requis
    """
    current_role = st.session_state.get("role", "user")
    return current_role == required_role

def is_admin():
    """
    Vérifie si l'utilisateur est administrateur
    Returns:
        bool: True si administrateur
    """
    return has_role("admin")

def require_admin():
    """
    Exige que l'utilisateur soit administrateur
    """
    if not is_admin():
        st.error("Accès refusé. Cette fonctionnalité nécessite des droits d'administrateur.")
        st.stop()

def get_user_role_display():
    """
    Obtient le nom d'affichage du rôle de l'utilisateur
    Returns:
        str: Nom d'affichage du rôle
    """
    role = st.session_state.get("role", "user")
    return ROLES.get(role, role.title())

def can_access_advanced_features():
    """
    Vérifie si l'utilisateur peut accéder aux fonctionnalités avancées
    Returns:
        bool: True si accès autorisé
    """
    return is_admin()

def get_accessible_pages():
    """
    Obtient la liste des pages accessibles selon le rôle
    Returns:
        list: Liste des pages accessibles
    """
    base_pages = ["prediction", "single_prediction"]
    
    if is_admin():
        admin_pages = ["data_analysis", "model_overview"]
        return base_pages + admin_pages
    
    return base_pages
