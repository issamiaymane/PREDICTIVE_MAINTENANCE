#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Configuration de la base de données MongoDB
"""

import streamlit as st
from pymongo import MongoClient
from config.settings import MONGODB_CONFIG

@st.cache_resource
def init_mongodb():
    """
    Initialise la connexion MongoDB
    Returns:
        tuple: (collection, connected_status)
    """
    try:
        client = MongoClient(
            f"mongodb://{MONGODB_CONFIG['host']}:{MONGODB_CONFIG['port']}/",
            serverSelectionTimeoutMS=MONGODB_CONFIG['timeout']
        )
        client.server_info()
        db = client[MONGODB_CONFIG['database']]
        collection = db[MONGODB_CONFIG['collection']]
        return collection, True
    except Exception as e:
        st.error(f"Erreur de connexion MongoDB: {e}")
        return None, False

def get_database_connection():
    """
    Obtient une connexion à la base de données
    Returns:
        MongoClient: Client MongoDB connecté
    """
    try:
        client = MongoClient(
            f"mongodb://{MONGODB_CONFIG['host']}:{MONGODB_CONFIG['port']}/",
            serverSelectionTimeoutMS=MONGODB_CONFIG['timeout']
        )
        return client
    except Exception as e:
        st.error(f"Erreur de connexion MongoDB: {e}")
        return None
