#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Page de connexion
"""

import streamlit as st
from auth.authentication import check_credentials, login_user
from ui.components import snrt_colors_display

def login_page(users_collection, db_connected):
    """
    Affiche la page de connexion
    Args:
        users_collection: Collection MongoDB des utilisateurs
        db_connected (bool): État de la connexion à la base de données
    """
    if not db_connected:
        st.error("❌ Erreur de connexion à la base de données MongoDB")
        st.info("Veuillez vérifier que MongoDB est lancé et accessible sur localhost:27017")
        st.stop()
    
    st.markdown("""
    <div>
        <div class="snrt-colors">
            <div class="color-dot color-blue"></div>
            <div class="color-dot color-red"></div>
            <div class="color-dot color-orange"></div>
            <div class="color-dot color-green"></div>
        </div>
        <div class="login">
        <h1>🔐 Connexion</h1>
        <p class="snrt-subtitle">Système de Maintenance Prédictive Intelligente</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    username = st.text_input("Nom d'utilisateur")
    password = st.text_input("Mot de passe", type="password")
    login_btn = st.button("Se connecter")

    if login_btn:
        if username and password:
            valid, user = check_credentials(username, password, users_collection)
            if valid:
                login_user(user)
                st.success(f"✅ Bienvenue {user['name']}!")
                st.rerun()
            else:
                st.error("❌ Nom d'utilisateur ou mot de passe incorrect")
        else:
            st.warning("⚠️ Veuillez remplir tous les champs")
