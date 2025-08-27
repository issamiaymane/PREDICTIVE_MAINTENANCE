#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gestion de l'authentification
"""

import streamlit as st
import bcrypt
from config.settings import ROLES

def check_credentials(username, password, users_collection):
    """
    Vérifie les credentials de l'utilisateur
    Args:
        username (str): Nom d'utilisateur
        password (str): Mot de passe
        users_collection: Collection MongoDB des utilisateurs
    Returns:
        tuple: (is_valid, user_data)
    """
    if users_collection is None:
        return False, None
    
    user = users_collection.find_one({"username": username})
    if user:
        if bcrypt.checkpw(password.encode(), user["password"].encode()):
            return True, user
    return False, None

def login_user(user_data):
    """
    Connecte l'utilisateur en définissant les variables de session
    Args:
        user_data (dict): Données de l'utilisateur
    """
    st.session_state.authenticated = True
    st.session_state.username = user_data["username"]
    st.session_state.name = user_data["name"]
    st.session_state.role = user_data.get("role", "user")

def logout_user():
    """
    Déconnecte l'utilisateur en supprimant les variables de session
    """
    for key in list(st.session_state.keys()):
        del st.session_state[key]

def is_authenticated():
    """
    Vérifie si l'utilisateur est authentifié
    Returns:
        bool: True si authentifié, False sinon
    """
    return st.session_state.get("authenticated", False)

def get_current_user():
    """
    Obtient les informations de l'utilisateur actuel
    Returns:
        dict: Informations de l'utilisateur ou None
    """
    if is_authenticated():
        return {
            "username": st.session_state.get("username"),
            "name": st.session_state.get("name"),
            "role": st.session_state.get("role", "user")
        }
    return None

def require_authentication():
    """
    Décorateur pour exiger l'authentification
    """
    if not is_authenticated():
        st.error("Veuillez vous connecter pour accéder à cette page.")
        st.stop()
